"""
全局配置
========
所有路径、超参数和设备选择集中在这里管理。
本作业为说话者分类（Speaker Classification），共 600 类。
采用 step-based 训练（与参考代码一致），非 epoch-based。
"""

import os
import torch

# ─── 设备选择 ──────────────────────────────────────────────────────
# torch.cuda.is_available(): 检测是否有可用的 NVIDIA GPU（CUDA）
# torch.backends.mps: Apple Silicon 的 Metal Performance Shaders
# getattr 安全检测: 旧版 PyTorch 可能没有 mps 模块
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and \
        torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─── 路径配置 ──────────────────────────────────────────────────────
# os.path: 跨平台的路径拼接工具
# __file__: 当前文件路径，向上两级定位项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录：包含 metadata.json, testdata.json, mapping.json, uttr-*.pt
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Dataset")

# 输出目录（保存模型权重、CSV 提交文件、训练曲线图）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ─── 数据集常量 ────────────────────────────────────────────────────
N_MELS = 40                 # mel-spectrogram 的特征维度（每帧 40 维）
SEGMENT_LEN = 128           # 训练时每个样本切出的时间步长度（参考代码默认 128）
VAL_RATIO = 0.1             # 训练/验证集划分比例：90% 训练，10% 验证

# ─── 训练超参数（step-based，与参考代码一致）──────────────────────
BATCH_SIZE = 32             # 批次大小（3070Ti 8G 显存）
N_WORKERS = 8               # DataLoader 并行进程数（云端 Linux 可安全使用）
WARMUP_STEPS = 1000         # 学习率线性预热的步数（Transformer 训练常用）
SAVE_STEPS = 10000          # 每隔多少步保存一次模型

LEARNING_RATE = 1e-3        # 初始学习率（warmup 后达到此值）

# ─── 各等级总训练步数 ──────────────────────────────────────────────
# 参考代码 total_steps=70000，约对应 Simple baseline
TOTAL_STEPS_SIMPLE = 70000    # Simple 等级
VALID_STEPS_SIMPLE = 2000     # Simple 每隔多少步验证一次

TOTAL_STEPS_MEDIUM = 70000    # Medium 等级（调参后相同样本量即可）
VALID_STEPS_MEDIUM = 2000

TOTAL_STEPS_HARD = 100000     # Hard 等级（Conformer 收敛更慢，增加步数）
VALID_STEPS_HARD = 2000


# ─── 优化器与正则化 ────────────────────────────────────────────────
WEIGHT_DECAY = 1e-4         # AdamW 的解耦权重衰减系数