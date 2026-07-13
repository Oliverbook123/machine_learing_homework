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

# 数据目录：包含 uttr-*.pt 特征文件
# metadata.json 和 mapping.json 在此目录，testdata.json 在 ml2021spring-hw4 子目录
# 注意: 本地从 Dropbox 下载后 mapping.json/metadata.json 在 Dataset/ 下，
#       testdata.json 在 data/ml2021spring-hw4/ 下
#       云端若数据全部在 data/ml2021spring-hw4/ 下，可改为:
#       DATA_DIR = os.path.join(DATA_ROOT, "ml2021spring-hw4")
#       或设置环境变量 HW4_DATA_DIR 覆盖
_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
if os.path.exists(os.path.join(_DATA_ROOT, "Dataset", "mapping.json")):
    DATA_DIR = os.path.join(_DATA_ROOT, "Dataset")
else:
    # 回退: mapping.json 等在 ml2021spring-hw4 子目录下
    DATA_DIR = os.path.join(_DATA_ROOT, "ml2021spring-hw4")

# 输出目录（保存模型权重、CSV 提交文件、训练曲线图）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ─── 数据集常量 ────────────────────────────────────────────────────
N_MELS = 40                 # mel-spectrogram 的特征维度（每帧 40 维）
# 改进: 128→256。更长片段提供更多语音上下文，说话者特征更完整
# 旧值: SEGMENT_LEN = 128（参考代码默认）
SEGMENT_LEN = 256           # 训练时每个样本切出的时间步长度
VAL_RATIO = 0.1             # 训练/验证集划分比例：90% 训练，10% 验证

# ─── 训练超参数（step-based，与参考代码一致）──────────────────────
# 改进: 32→16。d_model 增大后显存占用上升，需降低 batch 防 OOM
# 旧值: BATCH_SIZE = 32
BATCH_SIZE = 16             # 批次大小（3070Ti 8G 显存）
N_WORKERS = 8               # DataLoader 并行进程数（云端 Linux 可安全使用）
# 改进: 1000→5000。d_model 增大后梯度方差更大，需更长 warmup 稳定训练
# 旧值: WARMUP_STEPS = 1000
WARMUP_STEPS = 5000         # 学习率线性预热的步数（Transformer 训练常用）
SAVE_STEPS = 10000          # 每隔多少步保存一次模型

# 改进: 1e-3→5e-4。d_model 从 80→256 增大，lr 应按 1/√d_model 缩放
# 旧值: LEARNING_RATE = 1e-3（d_model=80 时适用）
LEARNING_RATE = 5e-4        # 初始学习率（warmup 后达到此值）

# ─── 各等级总训练步数 ──────────────────────────────────────────────
# 参考代码 total_steps=70000，约对应 Simple baseline
# 改进: Simple 70k→70k（保持，容量提升后同样步数即可达标）
TOTAL_STEPS_SIMPLE = 70000    # Simple 等级
VALID_STEPS_SIMPLE = 2000     # Simple 每隔多少步验证一次

# 改进: Medium 70k→100000（模型变大后需更多步数收敛）
# 旧值: TOTAL_STEPS_MEDIUM = 70000
TOTAL_STEPS_MEDIUM = 100000   # Medium 等级
VALID_STEPS_MEDIUM = 2000

# 改进: Hard 100k→150000（Conformer 容量增大后需更多步数）
# 旧值: TOTAL_STEPS_HARD = 100000
TOTAL_STEPS_HARD = 150000     # Hard 等级（Conformer 收敛更慢，增加步数）
VALID_STEPS_HARD = 2000


# ─── 优化器与正则化 ────────────────────────────────────────────────
WEIGHT_DECAY = 1e-4         # AdamW 的解耦权重衰减系数

# 改进: 新增标签平滑。600 类分类任务中标签平滑有助于泛化
# 旧值: 无（CrossEntropyLoss 无 label_smoothing 参数）
LABEL_SMOOTHING = 0.1       # 标签平滑系数，防止对训练标签过度自信