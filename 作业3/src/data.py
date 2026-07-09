"""
数据加载与预处理
=================
包含 FoodDataset 定义、数据增强/预处理变换、以及 DataLoader 创建函数。
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    LABELED_PER_CLASS, DEVICE,
)


# ═══════════════════════════════════════════════════════════════════
# 1. 自定义数据集
# ═══════════════════════════════════════════════════════════════════

class FoodDataset(Dataset):
    """
    PyTorch Dataset 的封装，用于加载 food-11 的图片和标签。

    文件名格式：
      - 训练/验证集: {class_id}_{image_id}.jpg（如 "0_0.jpg"）
      - 测试集:      {image_id}.jpg（如 "0000.jpg"）
    """

    def __init__(self, image_dir, transform=None, labeled_only=True):
        """
        参数:
            image_dir (str):           图片目录
            transform (callable, 可选):  图像预处理/增强函数
            labeled_only (bool):        是否只加载有标签的图片
        """
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []  # [(文件名, 标签), ...]，标签 -1 表示无标签

        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            parts = fname.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                cls = int(parts[0])
                img_id = int(parts[1].split('.')[0])

                if img_id < LABELED_PER_CLASS:
                    self.samples.append((fname, cls))          # 有标签
                elif not labeled_only:
                    self.samples.append((fname, -1))           # 无标签
            else:
                # 测试集: 无类别前缀
                self.samples.append((fname, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class FileListDataset(Dataset):
    """
    直接从文件路径和标签列表构建的数据集。
    每张图片在 __getitem__ 时即时加载并应用 transform。
    用于半监督学习中合并有标签 + 伪标签数据（只存路径，不占内存）。
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        参数:
            image_paths (list[str]):  图片文件的完整路径列表
            labels (list[int]):       对应的标签列表
            transform (callable, 可选): 数据增强/预处理函数
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ═══════════════════════════════════════════════════════════════════
# 2. 数据增强 / 预处理变换
# ═══════════════════════════════════════════════════════════════════

# ---- 基本预处理（无增强，用于 Easy 等级） ----
# T.Resize:    缩放图片到统一尺寸
# T.ToTensor:  PIL Image → Tensor (C×H×W)，像素值 [0,1]
# T.Normalize: 每个通道标准化，使用 ImageNet 统计量
train_transform_basic = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# ---- 数据增强（用于 Medium / Hard 等级的有标签训练） ----
# T.RandomHorizontalFlip:  50% 概率水平翻转
# T.RandomRotation:        随机旋转 ±15°
# T.ColorJitter:           亮度/对比度/饱和度/色调随机微调
# T.RandomResizedCrop:     随机裁剪再缩放，scale 控制裁剪面积比例
train_transform_augmented = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# ---- 更强的增强（用于半监督阶段，防止伪标签过拟合） ----
train_transform_semi = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# ---- 验证/测试：不做数据增强 ----
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════
# 3. 创建 DataLoader
# ═══════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size=None, num_workers=None):
    """创建三个等级所需的全部 DataLoader。

    返回:
        train_labeled_loader:  有标签训练数据
        train_aug_loader:      有标签训练数据（带数据增强）
        val_loader:            验证数据
        test_loader:           测试数据
        test_dataset:          测试集 Dataset（用于生成提交文件）
    """
    bs = batch_size or BATCH_SIZE
    nw = num_workers or NUM_WORKERS
    pin = DEVICE.type == "cuda"

    # 有标签训练（无增强）
    train_labeled_dataset = FoodDataset(
        TRAIN_DIR, transform=train_transform_basic, labeled_only=True
    )
    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin,
    )

    # 有标签训练（带数据增强）
    train_aug_dataset = FoodDataset(
        TRAIN_DIR, transform=train_transform_augmented, labeled_only=True
    )
    train_aug_loader = DataLoader(
        train_aug_dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin,
    )

    # 验证集
    val_dataset = FoodDataset(VAL_DIR, transform=val_transform, labeled_only=True)
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    # 测试集
    test_dataset = FoodDataset(TEST_DIR, transform=val_transform, labeled_only=False)
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    return (train_labeled_loader, train_aug_loader,
            val_loader, test_loader, test_dataset)


def get_unlabeled_paths(image_dir=None, labeled_per_class=None):
    """返回所有无标签图片的完整路径和占位标签列表。

    DataLoader 不能直接处理 PIL Image 的多进程序列化，所以这里只返回路径。
    在具体使用时手动加载图片。
    """
    image_dir = image_dir or TRAIN_DIR
    lpc = labeled_per_class or LABELED_PER_CLASS

    paths, labels = [], []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = fname.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            img_id = int(parts[1].split('.')[0])
            if img_id >= lpc:
                paths.append(os.path.join(image_dir, fname))
                labels.append(-1)
    return paths, labels


def get_labeled_paths(image_dir=None, labeled_per_class=None):
    """返回所有有标签图片的完整路径和真实标签。"""
    image_dir = image_dir or TRAIN_DIR
    lpc = labeled_per_class or LABELED_PER_CLASS

    paths, labels = [], []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = fname.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            cls = int(parts[0])
            img_id = int(parts[1].split('.')[0])
            if img_id < lpc:
                paths.append(os.path.join(image_dir, fname))
                labels.append(cls)
    return paths, labels
