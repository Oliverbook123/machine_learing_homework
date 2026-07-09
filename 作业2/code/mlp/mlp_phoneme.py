"""
作业2 - 任务2-1：音素分类（Phoneme Classification）
使用 MLP（多层感知机）实现逐帧音素预测
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# ============================================================
# 配置区
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/timit_11/timit_11"
OUT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 256        # 每批训练样本数
EPOCHS = 12             # 训练轮数
SEED = 42               # 随机种子，保证可复现
VALID_RATIO = 0.1       # 验证集比例（从训练集中划分）

# 学习率调度参数
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3      # Dropout 丢弃概率，防止过拟合

# 是否使用 GPU（如有）
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 1. 加载数据
# ============================================================
# np.load: 加载 .npy 格式的 numpy 数组文件
# train_11.npy 中每个样本是 429 维（11帧 × 39维MFCC）
# train_label_11.npy 中每个元素是对应中心帧的音素标签（字符串形式，需转整数）
train_x = np.load(DATA_DIR / "train_11.npy").astype(np.float32)
train_y = np.load(DATA_DIR / "train_label_11.npy")
test_x = np.load(DATA_DIR / "test_11.npy").astype(np.float32)

# 标签从字符串转换为整数（如 '0'、'1'... → 0、1...）
train_y = train_y.astype(np.int64)

num_train = int(len(train_x) * (1 - VALID_RATIO))

# ============================================================
# 2. 构建 DataLoader
# ============================================================
# TensorDataset: 将特征和标签打包为 (x, y) 对，方便 DataLoader 批量加载
# DataLoader: 自动进行批量采样、打乱（shuffle=True 仅在训练时使用）
train_dataset = TensorDataset(
    torch.from_numpy(train_x[:num_train]),
    torch.from_numpy(train_y[:num_train]),
)
valid_dataset = TensorDataset(
    torch.from_numpy(train_x[num_train:]),
    torch.from_numpy(train_y[num_train:]),
)

# num_workers: 并行加载数据的进程数；pin_memory: 将数据锁定在内存中加快 GPU 传输
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(test_x)),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
)

print(f"训练集: {len(train_dataset)} 样本  验证集: {len(valid_dataset)} 样本  测试集: {len(test_x)} 样本")

# ============================================================
# 3. 模型定义 — MLP + BatchNorm + Dropout
# ============================================================
class MLPClassifier(nn.Module):
    """
    多层感知机音素分类器

    使用 nn.Sequential 简洁地堆叠各层:
      输入(429) → Linear(429→512) + BN + ReLU + Dropout
               → Linear(512→256) + BN + ReLU + Dropout
               → Linear(256→128) + BN + ReLU + Dropout
               → Linear(128→39)  输出类别 logits
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        # nn.Sequential: 有序容器，输入按定义顺序依次流过每一层
        # 前向传播只需 self.network(x) 一步完成，无需手动循环
        self.network = nn.Sequential(
            # ========= 第一层: 429 → 512 =========
            # nn.Linear(429, 512): 全连接层，将 429 维输入线性映射到 512 维
            nn.Linear(429, 512),
            # nn.BatchNorm1d(512): 批归一化，对 512 维输出的每个维度在当前 batch 内做标准化
            # 公式: (x - μ_batch) / σ_batch * γ + β，γ/β 是可学习参数
            nn.BatchNorm1d(512),
            # nn.ReLU: 激活函数 f(x)=max(0, x)，引入非线性
            nn.ReLU(),
            # nn.Dropout(0.3): 训练时随机丢弃 30% 的神经元输出（置零），防止过拟合
            nn.Dropout(dropout),

            # ========= 第二层: 512 → 256 =========
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ========= 第三层: 256 → 128 =========
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ========= 输出层: 128 → 39 =========
            # 输出层不加 BN/激活：CrossEntropyLoss 内部会做 softmax，
            # 我们需要 raw logits 来保持数值稳定性
            nn.Linear(128, 39),
        )

    def forward(self, x):
        # 数据一次性流过 Sequential 中定义的所有层
        return self.network(x)


# 实例化模型并移到指定设备（GPU/CPU）
model = MLPClassifier(dropout=DROPOUT_RATE).to(device)

# ============================================================
# 4. 损失函数 & 优化器 & 学习率调度器
# ============================================================
# nn.CrossEntropyLoss: 交叉熵损失，适用于多分类
# 内部自动对 logits 做 LogSoftmax + NLLLoss
# 注意：不要对输入先手动做 softmax！
criterion = nn.CrossEntropyLoss()

# nn.Adam: Adam 优化器，自适应学习率，比 SGD 更适合直接使用
# weight_decay: L2 正则化，约束权重不变得太大，防过拟合
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# optim.lr_scheduler.CosineAnnealingLR: 余弦退火学习率调度
# 从初始 lr 平滑衰减到 0，有助于训练后期稳定收敛
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============================================================
# 5. 训练循环
# ============================================================
print("\n开始训练...")
best_val_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # --- 训练阶段 ---
    model.train()  # 切换到训练模式（启用 BatchNorm 统计和 Dropout）
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # optimizer.zero_grad(): 清空上一轮累积的梯度
        optimizer.zero_grad()

        # model(xb): 前向传播，得到模型预测 logits
        logits = model(xb)

        # criterion(logits, yb): 计算交叉熵损失
        loss = criterion(logits, yb)

        # loss.backward(): 反向传播，自动计算所有参数的梯度 ∂loss/∂θ
        loss.backward()

        # optimizer.step(): 根据梯度更新所有参数
        optimizer.step()

        # 统计训练集准确率
        preds = logits.argmax(dim=1)  # argmax: 取概率最大的类别索引
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)
        train_loss_sum += loss.item() * yb.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # --- 验证阶段 ---
    model.eval()  # 切换到评估模式（禁用 Dropout，BatchNorm 用移动平均）
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    # torch.no_grad(): 禁用梯度计算，节省显存、加速推理
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
            val_loss_sum += loss.item() * yb.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    # 更新学习率
    scheduler.step()

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()
        torch.save(best_state, OUT_DIR / "best_model.pt")

    print(
        f"Epoch [{epoch:2d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

print(f"\n最佳验证准确率: {best_val_acc:.4f}")

# ============================================================
# 6. 加载最优模型并进行后处理预测
# ============================================================
# 加载训练过程中验证集准确率最好的模型权重
model.load_state_dict(torch.load(OUT_DIR / "best_model.pt"))
model.eval()

# 收集所有测试集预测结果
all_preds = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

# 拼接各批次的预测结果，得到完整的测试集预测
all_preds = np.concatenate(all_preds, axis=0)

# ----------------------------------------------------------
# 后处理：滑动窗口平滑（对音素预测做简单平滑）
# 原理：同一个音素通常跨越多帧，孤立的预测很可能是噪声
# 方法：用 uniform_filter1d 做窗口大小为 7 的均值滤波
#       mode='nearest' 表示边界处用最近的有效值填充
# ----------------------------------------------------------
from scipy.ndimage import uniform_filter1d

smooth_kernel = 7  # 窗口大小
raw_preds = all_preds.copy()
# uniform_filter1d: 一维滑动均值滤波，等价于卷积 [1/7]*7
# mode='nearest': 边界处用最外侧值延伸，避免截断
all_preds = uniform_filter1d(all_preds.astype(float), size=smooth_kernel, mode='nearest').round().astype(int)

print(f"后处理前/后差异样本数: {(raw_preds != all_preds).sum()} / {len(all_preds)}")

# ============================================================
# 7. 生成提交文件
# ============================================================
# 格式: 两列 —— Id（行号）和 Class（预测类别）
import csv

submission_path = OUT_DIR / "submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Class"])
    for idx, pred in enumerate(all_preds):
        writer.writerow([idx, pred])

print(f"提交文件已保存: {submission_path}")
