"""
训练与评估工具
==============
参照参考代码实现 step-based 训练，包含:
  - get_cosine_schedule_with_warmup:  warmup + cosine 退火学习率调度
  - model_fn:     单 batch 前向 + 损失 + 精度
  - valid:        验证集评估
  - train_model:  step-based 训练主循环
  - predict:      推理并生成提交 CSV
"""

import os
import math
import csv
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import DEVICE, N_WORKERS, DATA_DIR, LEARNING_RATE, WEIGHT_DECAY
from src.data import InferenceDataset, inference_collate_batch
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════
# 1. 学习率调度器（来自参考代码）
# ═══════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps,
    num_cycles=0.5, last_epoch=-1,
):
    """
    Warmup + Cosine 退火学习率调度。

    - 训练前 num_warmup_steps 步：学习率线性从 0 增长到初始 lr
    - 之后按余弦函数从初始 lr 衰减到 0

    参数:
        optimizer:           优化器
        num_warmup_steps:    预热步数
        num_training_steps:  总训练步数
        num_cycles:          余弦周期数（0.5 = 半个余弦）
        last_epoch:          恢复训练时的上轮索引

    返回:
        LambdaLR 实例
    """

    def lr_lambda(current_step):
        # Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine 退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    # LambdaLR: 用 lambda 函数动态计算每步的学习率倍率
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ═══════════════════════════════════════════════════════════════════
# 2. 模型前向函数（来自参考代码）
# ═══════════════════════════════════════════════════════════════════

def model_fn(batch, model, criterion, device):
    """单 batch 前向 + 计算损失和精度。

    参数:
        batch:     (mels, labels) 元组
        model:     模型
        criterion: 损失函数
        device:    训练设备

    返回:
        (loss, accuracy)
    """
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)
    loss = criterion(outs, labels)

    # argmax(1): 沿 dim=1 取概率最大的类别索引
    preds = outs.argmax(1)
    # (preds == labels).float(): bool → 0/1，mean 即精度
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


# ═══════════════════════════════════════════════════════════════════
# 3. 验证函数（来自参考代码）
# ═══════════════════════════════════════════════════════════════════

def valid(dataloader, model, criterion, device):
    """在验证集上计算平均精度。

    参数:
        dataloader: 验证 DataLoader
        model:      模型
        criterion:  损失函数
        device:     设备

    返回:
        平均精度
    """
    # model.eval(): 切到评估模式（关 Dropout，BN 用累计统计量）
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
        running_loss += loss.item()
        running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )
    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


# ═══════════════════════════════════════════════════════════════════
# 4. 训练主循环（step-based，参照参考代码）
# ═══════════════════════════════════════════════════════════════════

def train_model(model, train_loader, valid_loader,
                total_steps, valid_steps, warmup_steps, save_steps,
                device=DEVICE, model_name="model",
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                save_dir=None):
    """
    Step-based 训练循环。

    参数:
        model:          待训练模型
        train_loader:   训练 DataLoader
        valid_loader:   验证 DataLoader
        total_steps:    总训练步数
        valid_steps:    每隔多少步验证
        warmup_steps:   warmup 步数
        save_steps:     每隔多少步保存
        device:         训练设备
        model_name:      模型名称（用于保存文件命名）
        learning_rate:  初始学习率
        weight_decay:   权重衰减
        save_dir:       保存目录

    返回:
        model:       训练好的模型（最优权重）
        best_acc:    最优验证精度
    """
    from src.config import OUTPUT_DIR
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    # nn.CrossEntropyLoss: 交叉熵损失（内含 softmax）
    criterion = nn.CrossEntropyLoss()

    # AdamW: Adam + 解耦权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    # Warmup + Cosine 调度
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_accuracy = -1.0
    best_state_dict = None

    # train_loader 可以无限迭代：迭代完自动重新开始
    train_iterator = iter(train_loader)

    print(f"\n开始训练 {model_name} ...")
    print(f"  总步数: {total_steps}, 验证步数间隔: {valid_steps}, "
          f"warmup: {warmup_steps}")
    print("=" * 60)

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # 获取下一个 batch；迭代完整个训练集则重置迭代器
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 日志
        pbar.update()
        pbar.set_postfix(
            loss=f"{loss.item():.2f}",
            accuracy=f"{accuracy.item():.2f}",
            step=step + 1,
        )

        # 验证
        if (step + 1) % valid_steps == 0:
            pbar.close()
            valid_accuracy = valid(valid_loader, model, criterion, device)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                # 保存最优模型
                path = os.path.join(save_dir, f"{model_name}_best.pth")
                torch.save(best_state_dict, path)
                pbar.write(f"Step {step + 1}, best model saved. "
                           f"(accuracy={best_accuracy:.4f})")

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # 定期保存
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(best_state_dict, path)
            pbar.write(f"Step {step + 1}, best model saved. "
                       f"(accuracy={best_accuracy:.4f})")

    pbar.close()

    # 加载最优权重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\n最优验证精度: {best_accuracy:.4f}")
    return model, best_accuracy


# ═══════════════════════════════════════════════════════════════════
# 5. 推理与提交
# ═══════════════════════════════════════════════════════════════════

def predict(model, data_dir=DATA_DIR, output_path=None,
            model_name="model", save_dir=None):
    """
    对测试集推理并生成提交 CSV。

    参数:
        model:       训练好的模型
        data_dir:     数据目录（含 testdata.json 和 uttr-*.pt）
        output_path:  输出 CSV 路径
        model_name:   模型名称（用于默认输出文件名）
        save_dir:     输出目录

    返回:
        results: list of [feat_path, speaker_string_id]
    """
    import json
    from src.config import OUTPUT_DIR
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    output_path = output_path or os.path.join(save_dir, f"{model_name}_submission.csv")

    model.eval()

    # 加载 mapping.json（数字 ID → 说话者字符串 ID）
    mapping_path = os.path.join(data_dir, "mapping.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    dataset = InferenceDataset(data_dir)
    # batch_size=1: 推理时逐条处理，避免长度不一致导致 stack 失败
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=N_WORKERS, collate_fn=inference_collate_batch,
    )

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            mels = mels.to(device=DEVICE)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                # mapping["id2speaker"]: {"0": "id10001", "1": "id10005", ...}
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    # 写入 CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"提交文件已生成: {output_path}")
    return results