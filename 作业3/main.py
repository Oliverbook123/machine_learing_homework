"""
机器学习作业3 - 图像分类 (Food-11)
===================================
主入口脚本，按三个等级依次执行。

使用方法:
    uv run python main.py

等级说明:
  - Easy:   基础 CNN，仅使用有标签数据
  - Medium: 数据增强 + ImprovedCNN
  - Hard:   半监督学习 Pseudo-Labeling，利用无标签数据
"""

import os
import sys

# 确保项目根目录在 Python 路径中，使得 `import src` 可以工作
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.config import (
    DEVICE,
    NUM_EPOCHS_BASIC, NUM_EPOCHS_MEDIUM, NUM_EPOCHS_HARD,
    LEARNING_RATE,
)
from src.data import get_dataloaders
from src.models import BasicCNN, ImprovedCNN
from src.train import (
    train_model, predict, generate_submission, plot_history,
)
from src.semi_supervised import semi_supervised_training


def main():
    print("=" * 60)
    print("机器学习作业3 - 图像分类 (Food-11)")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[0/3] 加载数据...")
    (train_loader_labeled, train_aug_loader,
     val_loader, test_loader, test_dataset) = get_dataloaders()

    print(f"  有标签训练集: {len(train_loader_labeled.dataset)} 张")
    print(f"  数据增强训练集: {len(train_aug_loader.dataset)} 张")
    print(f"  验证集:         {len(val_loader.dataset)} 张")
    print(f"  测试集:         {len(test_loader.dataset)} 张")

    # ═════════════════════════════════════════════════════════════
    # LEVEL 1: EASY
    # ═════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("等级1: EASY - 基础 CNN（仅使用有标签数据，无数据增强）")
    print("=" * 60)

    model_easy = BasicCNN(num_classes=11).to(DEVICE)

    model_easy, history_easy = train_model(
        model_easy, train_loader_labeled, val_loader,
        num_epochs=NUM_EPOCHS_BASIC, learning_rate=LEARNING_RATE,
        device=DEVICE, model_name="basic_cnn",
    )

    preds_easy = predict(model_easy, test_loader)
    generate_submission(preds_easy, test_dataset, "submission_easy.csv")

    plot_history(history_easy, "Easy_BasicCNN")

    # ═════════════════════════════════════════════════════════════
    # LEVEL 2: MEDIUM — 数据增强 + ImprovedCNN
    # ═════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("等级2: MEDIUM - 数据增强 + ImprovedCNN")
    print("=" * 60)

    model_medium = ImprovedCNN(num_classes=11).to(DEVICE)

    model_medium, history_medium = train_model(
        model_medium, train_aug_loader, val_loader,
        num_epochs=NUM_EPOCHS_MEDIUM, learning_rate=LEARNING_RATE,
        device=DEVICE, model_name="improved_cnn",
    )

    preds_medium = predict(model_medium, test_loader)
    generate_submission(preds_medium, test_dataset, "submission_medium.csv")

    plot_history(history_medium, "Medium_ImprovedCNN")

    # ── 直接使用 ImprovedCNN 用于 Hard 等级 ──
    best_medium_acc = max(history_medium['val_acc'])
    print(f"\n→ 使用 ImprovedCNN (acc={best_medium_acc:.2f}%) 进入 Hard 等级")

    model_hard = ImprovedCNN(num_classes=11).to(DEVICE)
    model_hard.load_state_dict(
        torch.load("output/improved_cnn_best.pth", map_location=DEVICE)
    )

    # ═════════════════════════════════════════════════════════════
    # LEVEL 3: HARD — 半监督学习
    # ═════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("等级3: HARD - 半监督学习 (Pseudo-Labeling)")
    print("=" * 60)

    model_hard, hard_best_acc = semi_supervised_training(
        model_hard, val_loader,
        num_epochs=NUM_EPOCHS_HARD, learning_rate=LEARNING_RATE,
        device=DEVICE, model_name="semi_final",
    )

    preds_hard = predict(model_hard, test_loader)
    generate_submission(preds_hard, test_dataset, "submission_hard.csv")

    # ═════════════════════════════════════════════════════════════
    # 最终总结
    # ═════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("最终结果总结")
    print("=" * 60)
    print(f"  Easy   - BasicCNN:                {max(history_easy['val_acc']):.2f}%")
    print(f"  Medium - ImprovedCNN:             {best_medium_acc:.2f}%")
    if hard_best_acc:
        print(f"  Hard   - 半监督 Pseudo-Label:    {hard_best_acc:.2f}%")
    else:
        print("  Hard   - 半监督 Pseudo-Label:    未完成")
    print("\n提交文件:")
    print("  - output/submission_easy.csv")
    print("  - output/submission_medium.csv")
    print("  - output/submission_hard.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
