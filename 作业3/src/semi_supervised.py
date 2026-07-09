"""
半监督学习（Pseudo-Labeling）
==============================
Hard 等级的核心：利用无标签数据提升模型性能。

方法：
1. 用现有的有标签数据模型对无标签数据做推理
2. 按置信度阈值筛选，把高置信度预测作为"伪标签"
3. 将有标签数据 + 伪标签数据合并，联合训练
4. 迭代多轮，逐步降低阈值纳入更多数据

参考: Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised
      Learning Method for Deep Neural Networks", ICML 2013 Workshop
"""

import os
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    TRAIN_DIR, DEVICE,
    BATCH_SIZE, NUM_CLASSES, LABELED_PER_CLASS,
    WEIGHT_DECAY, LABEL_SMOOTHING, OUTPUT_DIR,
)
from src.data import (
    FileListDataset,
    get_labeled_paths, get_unlabeled_paths,
    val_transform, train_transform_semi,
)
from src.train import evaluate


# ═══════════════════════════════════════════════════════════════════
# 伪标签生成
# ═══════════════════════════════════════════════════════════════════

def generate_pseudo_labels(model, unlabeled_paths, transform, device,
                           threshold=None):
    """
    对无标签图片做推理，按置信度阈值生成伪标签。

    手动分批处理（不依赖 DataLoader 的 collate），避免 PIL Image
    在多进程环境下的序列化问题。

    参数:
        model:             训练好的模型
        unlabeled_paths:   无标签图片路径列表
        transform:         预处理变换（Resize + ToTensor + Normalize）
        device:            计算设备
        threshold:         置信度阈值，只保留 max(softmax) > threshold 的预测

    返回:
        selected_paths:   高置信度样本的路径列表
        selected_labels:  对应的伪标签列表
    """
    model.eval()

    selected_paths = []
    selected_labels = []

    with torch.no_grad():
        for start in tqdm(range(0, len(unlabeled_paths), BATCH_SIZE),
                          desc="生成伪标签"):
            end = min(start + BATCH_SIZE, len(unlabeled_paths))
            batch_paths = unlabeled_paths[start:end]

            # 加载并预处理一批图片
            tensors = []
            for path in batch_paths:
                pil_img = Image.open(path).convert('RGB')
                tensors.append(transform(pil_img))
            batch_tensor = torch.stack(tensors).to(device)

            # 推理
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # 按阈值筛选
            for i, path in enumerate(batch_paths):
                if threshold is None or max_probs[i].item() > threshold:
                    selected_paths.append(path)
                    selected_labels.append(preds[i].item())

    if not selected_paths:
        print(f"⚠️  没有生成任何伪标签（阈值={threshold} 可能过高）")
        return None, None

    print(f"生成了 {len(selected_paths)} 个伪标签 (阈值={threshold})")
    return selected_paths, selected_labels


# ═══════════════════════════════════════════════════════════════════
# 半监督训练主流程（迭代式伪标签法）
# ═══════════════════════════════════════════════════════════════════

def semi_supervised_training(model, val_loader, num_epochs, learning_rate,
                              device=DEVICE, model_name="semi_final",
                              model_dir=None):
    """
    多轮伪标签迭代训练。

    流程:
      1. 用当前模型对无标签数据生成伪标签
      2. 合并有标签数据 + 伪标签数据
      3. 联合训练若干 epoch
      4. 回到步骤 1，重复 K 轮

    参数:
        model:          已训练好的初始模型
        val_loader:     验证集 DataLoader
        num_epochs:     总的训练轮数
        learning_rate:  学习率
        model_name:     模型保存前缀
        model_dir:      保存目录
    """
    save_dir = model_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("半监督学习阶段 (Pseudo-Labeling)")
    print("=" * 60)

    # ── 准备数据路径 ──
    labeled_paths, labeled_labels = get_labeled_paths()
    unlabeled_paths, _ = get_unlabeled_paths()
    print(f"\n有标签: {len(labeled_paths)} 张  |  无标签: {len(unlabeled_paths)} 张")

    # ── 超参数 ──
    num_iterations = 3            # 迭代轮数
    threshold_start = 0.95        # 初始置信度阈值
    threshold_end = 0.85          # 最终置信度阈值
    epochs_per_iter = max(1, num_epochs // num_iterations)  # 每轮训练的 epoch 数

    # 标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for iteration in range(num_iterations):
        print(f"\n{'─' * 50}")
        print(f"伪标签迭代 {iteration + 1}/{num_iterations}")

        # 动态调整阈值：越往后阈值越低，纳入更多数据
        if num_iterations > 1:
            thresh = threshold_start - (threshold_start - threshold_end) * \
                     iteration / (num_iterations - 1)
        else:
            thresh = threshold_start
        print(f"置信度阈值: {thresh:.2f}")

        # ── 生成伪标签 ──
        pseudo_paths, pseudo_labels = generate_pseudo_labels(
            model, unlabeled_paths, val_transform, device,
            threshold=thresh
        )

        if pseudo_paths is None or len(pseudo_paths) == 0:
            print("⚠️  无足够伪标签，提前结束")
            break

        # ── 合并有标签 + 伪标签 ──
        all_paths = labeled_paths + pseudo_paths
        all_labels = labeled_labels + pseudo_labels
        print(f"联合训练集: {len(all_paths)} 张 "
              f"(有标签: {len(labeled_paths)} + 伪标签: {len(pseudo_paths)})")

        # 使用 FileListDataset（只存路径，不提前加载图片到内存）
        # 配合 train_transform_semi（更强的数据增强）
        combined_dataset = FileListDataset(
            all_paths, all_labels, transform=train_transform_semi
        )
        combined_loader = DataLoader(
            combined_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0,
        )

        # ── 联合训练 ──
        # 用 AdamW + 较小学习率，防止破坏已有知识
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate * 0.5,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_iter
        )

        for epoch in range(1, epochs_per_iter + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(combined_loader,
                                       desc=f"半监督 Epoch {epoch}/{epochs_per_iter}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scheduler.step()

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())

            print(f"  Epoch {epoch:2d}: "
                  f"Train Loss={running_loss / len(combined_loader):.4f}  "
                  f"Train Acc={100.0 * correct / total:.2f}%  "
                  f"Val Acc={val_acc:.2f}%  (Best: {best_val_acc:.2f}%)")

        # 加载本轮最优权重，用于下一轮伪标签生成
        model.load_state_dict(best_wts)

    # ── 保存最终模型 ──
    model.load_state_dict(best_wts)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(best_wts, save_path)
    print(f"\n半监督模型已保存: {save_path}")
    print(f"最优验证准确率: {best_val_acc:.2f}%")

    return model, best_val_acc
