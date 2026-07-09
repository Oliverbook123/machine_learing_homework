"""
训练与评估工具
===============
包含训练循环、验证评估、推理预测、提交文件生成、训练曲线绘制。
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, OUTPUT_DIR, BATCH_SIZE,
    WEIGHT_DECAY, LABEL_SMOOTHING,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch（遍历一次全部训练数据）。"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="训练"):
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

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """在验证/测试集上评估模型（无梯度计算，禁用 Dropout/BatchNorm 训练行为）。"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="评估"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, num_epochs, learning_rate,
                device=DEVICE, model_name="model", weight_decay=None,
                label_smoothing=None, model_dir=None):
    """
    完整的训练循环。

    参数:
        weight_decay:    L2 正则化系数（默认从 config 获取）
        label_smoothing: 标签平滑系数（默认从 config 获取）
        model_dir:       模型保存目录（默认 OUTPUT_DIR）

    返回:
        model:    训练好的模型（验证集最优权重）
        history:  { 'train_loss', 'train_acc', 'val_loss', 'val_acc' }
    """
    wd = weight_decay if weight_decay is not None else WEIGHT_DECAY
    ls = label_smoothing if label_smoothing is not None else LABEL_SMOOTHING
    save_dir = model_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # nn.CrossEntropyLoss(label_smoothing): 交叉熵损失 + 标签平滑
    # 标签平滑防止模型对训练标签过于自信，缓解过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    # optim.AdamW: Adam + 解耦权重衰减，比普通 Adam+L2 更有效
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)

    # CosineAnnealingLR: 余弦退火学习率，从初始值逐渐降到 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, best_wts = 0.0, None
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    print(f"\n开始训练 {model_name} ...")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start
        print(f"  Epoch {epoch:2d}/{num_epochs}  "
              f"Train Loss={train_loss:.4f}  Acc={train_acc:.2f}%  "
              f"Val Acc={val_acc:.2f}%  (Best: {best_acc:.2f}%)  "
              f"LR={scheduler.get_last_lr()[0]:.6f}  "
              f"Time={elapsed:.1f}s")

    model.load_state_dict(best_wts)
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    torch.save(best_wts, save_path)
    print(f"\n模型已保存: {save_path}")
    print(f"最优验证准确率: {best_acc:.2f}%")

    return model, history


def predict(model, dataloader, device=DEVICE):
    """对测试集进行推理，返回预测标签列表。"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="预测"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


def generate_submission(predictions, test_dataset, filename="submission.csv",
                        model_dir=None):
    """
    生成 Kaggle 提交 CSV。

    格式:
        Id, Category
        0000, 3
        0001, 7
        ...
    """
    save_dir = model_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        f.write("Id, Category\n")
        for idx, pred in enumerate(predictions):
            fname = test_dataset.samples[idx][0]
            img_id = os.path.splitext(fname)[0]
            f.write(f"{img_id}, {pred}\n")

    print(f"提交文件已生成: {filepath}")


def plot_history(history, title, model_dir=None):
    """绘制训练历史曲线并保存图片。"""
    save_dir = model_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{title}_training_curve.png')
    plt.savefig(save_path)
    print(f"训练曲线已保存: {save_path}")
    plt.close()  # 关闭图形释放内存
