"""
机器学习作业4 - 说话者分类 (Speaker Classification)
===================================================
主入口脚本，按三个等级依次执行（step-based 训练，与参考代码一致）。

使用方法:
    uv run python main.py

等级说明:
  - Simple: 基础 Transformer (d_model=80, nhead=2, 2层)
  - Medium: 调参 Transformer (nhead=4, 4层) + 自注意力池化
  - Hard:   Conformer + 自注意力池化

设备: 自动检测 CUDA / MPS / CPU
"""

import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.config import (
    DEVICE, DATA_DIR, BATCH_SIZE, N_WORKERS,
    WARMUP_STEPS, SAVE_STEPS,
    TOTAL_STEPS_SIMPLE, VALID_STEPS_SIMPLE,
    TOTAL_STEPS_MEDIUM, VALID_STEPS_MEDIUM,
    TOTAL_STEPS_HARD, VALID_STEPS_HARD,
    LEARNING_RATE,
)
from src.data import get_dataloader
from src.models import Classifier, ClassifierV2, ConformerClassifier
from src.train import (
    train_model, predict,
)


def main():
    print("=" * 60)
    print("机器学习作业4 - 说话者分类 (Speaker Classification)")
    print(f"设备: {DEVICE}")
    if DEVICE.type == "cuda":
        # torch.cuda.get_device_name(0): 获取第 0 张 GPU 名称
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[0/3] 加载数据...")
    train_loader, valid_loader, speaker_num = get_dataloader(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, n_workers=N_WORKERS,
    )

    # ═══════════════════════════════════════════════════════════════
    # Simple 等级：基础 Transformer
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[1/3] Simple 等级: 基础 Transformer")
    print("=" * 60)

    # 与参考代码一致: d_model=80, nhead=2, dim_ff=256, 2层
    simple_model = Classifier(
        d_model=80, n_spks=speaker_num, dropout=0.1,
    )

    simple_model, simple_acc = train_model(
        simple_model, train_loader, valid_loader,
        total_steps=TOTAL_STEPS_SIMPLE,
        valid_steps=VALID_STEPS_SIMPLE,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        model_name="simple",
        learning_rate=LEARNING_RATE,
    )

    # 推理并生成提交文件
    predict(simple_model, data_dir=DATA_DIR, model_name="simple")

    # ═══════════════════════════════════════════════════════════════
    # Medium 等级：调参 Transformer + 自注意力池化
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[2/3] Medium 等级: 调参 Transformer + 自注意力池化")
    print("=" * 60)

    medium_model = ClassifierV2(
        d_model=80, n_spks=speaker_num, dropout=0.1,
    )

    medium_model, medium_acc = train_model(
        medium_model, train_loader, valid_loader,
        total_steps=TOTAL_STEPS_MEDIUM,
        valid_steps=VALID_STEPS_MEDIUM,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        model_name="medium",
        learning_rate=LEARNING_RATE,
    )

    predict(medium_model, data_dir=DATA_DIR, model_name="medium")

    # ═══════════════════════════════════════════════════════════════
    # Hard 等级：Conformer
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[3/3] Hard 等级: Conformer")
    print("=" * 60)

    hard_model = ConformerClassifier(
        d_model=80, n_spks=speaker_num, num_blocks=4,
        nhead=4, dim_ff=512, dropout=0.1,
    )

    hard_model, hard_acc = train_model(
        hard_model, train_loader, valid_loader,
        total_steps=TOTAL_STEPS_HARD,
        valid_steps=VALID_STEPS_HARD,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        model_name="hard",
        learning_rate=LEARNING_RATE,
    )

    predict(hard_model, data_dir=DATA_DIR, model_name="hard")

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("全部训练完成！")
    print(f"  Simple  最高验证精度: {simple_acc:.4f}")
    print(f"  Medium  最高验证精度: {medium_acc:.4f}")
    print(f"  Hard    最高验证精度: {hard_acc:.4f}")
    print("提交文件已保存到 output/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()