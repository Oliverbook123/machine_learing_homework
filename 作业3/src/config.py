"""
全局配置
=========
所有路径、超参数和设备选择集中在这里管理。
"""

import os
import torch

# ─── 设备选择 ──────────────────────────────────────────────────────
# torch.device: 自动选择最优硬件（CUDA GPU > Apple MPS > CPU）
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─── 路径配置 ──────────────────────────────────────────────────────
# 项目根目录（main.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "food-11")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "testing")

# 输出目录（保存模型权重、CSV 提交文件等）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ─── 数据集常量 ────────────────────────────────────────────────────
NUM_CLASSES = 11               # 食物类别数（0 ~ 10）
LABELED_PER_CLASS = 280        # 每个类别的前 N 张为有标签数据
NUM_UNLABELED = 6786           # 无标签图像总数（仅提示用）


# ─── 训练超参数 ────────────────────────────────────────────────────
BATCH_SIZE = 16                # 批次大小（224 图较大，减小防 OOM）
NUM_EPOCHS_BASIC = 30          # Easy 等级训练轮数
NUM_EPOCHS_MEDIUM = 40         # Medium 等级训练轮数
NUM_EPOCHS_HARD = 30           # Hard 等级训练轮数
LEARNING_RATE = 1e-3           # 初始学习率
IMAGE_SIZE = 224               # 统一缩放尺寸（保留更多细节，提升精度）
NUM_WORKERS = 0                # 数据加载并行数（云端设 0 避免多进程兼容问题）


# ─── 优化器与正则化 ────────────────────────────────────────────────
WEIGHT_DECAY = 5e-4            # L2 权重衰减系数，抑制过拟合
LABEL_SMOOTHING = 0.1          # 标签平滑系数（CrossEntropyLoss 参数）
