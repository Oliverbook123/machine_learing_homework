"""
作业2 - 任务2-1：音素分类 —— CNN（卷积神经网络）版本
输入: 429 维 MFCC 特征（11帧 × 39维）
方法: 重塑为 (39, 11)，将 39 种 MFCC 系数视为通道，11 帧视为时间轴，用 Conv1d 捕捉帧间局部模式
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# ============================================================
# 配置区
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/timit_11/timit_11"
OUT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42
VALID_RATIO = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

# 设置种子
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 1. 加载数据
# ============================================================
# 数据说明:
#   train_11.npy 的每一行是 429 维向量，代表 11 帧 × 39 维 MFCC
#   标签是对应中心帧（第 6 帧）的音素类别
#   CNN 视角：将 429 重塑为 (39, 11)，39 为通道数，11 为时间轴上的序列长度
train_x = np.load(DATA_DIR / "train_11.npy").astype(np.float32)
train_y = np.load(DATA_DIR / "train_label_11.npy").astype(np.int64)
test_x = np.load(DATA_DIR / "test_11.npy").astype(np.float32)

num_train = int(len(train_x) * (1 - VALID_RATIO))

# ============================================================
# 2. 构建 DataLoader
# ============================================================
train_dataset = TensorDataset(
    torch.from_numpy(train_x[:num_train]),
    torch.from_numpy(train_y[:num_train]),
)
valid_dataset = TensorDataset(
    torch.from_numpy(train_x[num_train:]),
    torch.from_numpy(train_y[num_train:]),
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(test_x)),
    batch_size=BATCH_SIZE, shuffle=False,
)

print(f"训练集: {len(train_dataset):,} 样本  验证集: {len(valid_dataset):,} 样本  测试集: {len(test_x):,} 样本")


# ============================================================
# 3. 模型定义 — CNN
# ============================================================
class CNNPhonemeClassifier(nn.Module):
    """
    CNN 音素分类器

    输入是 429 维平铺向量（11帧 × 39维MFCC），
    CNN 的第一步将其重塑为 (batch, 39, 11)：
      - 39 = 通道数（每种 MFCC 系数作为一个独立通道）
      - 11 = 序列长度（沿时间轴的 11 帧）

    用 Conv1d 沿时间维度做卷积，捕捉相邻帧之间的局部时序模式。

    架构:
      输入 (batch, 429) → reshape → (batch, 39, 11)
      → Conv1d(39→128, k=3) + BN + ReLU          # 提取帧间局部特征
      → Conv1d(128→256, k=3) + BN + ReLU          # 更高层抽象
      → AdaptiveAvgPool1d(1) → flatten → (256)    # 全局均值池化，聚合时间信息
      → Linear(256→128) + BN + ReLU + Dropout     # 分类头
      → Linear(128→39)                            # 输出 39 类 logits
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # === CNN 特征提取器 ===
        # nn.Conv1d(in_channels, out_channels, kernel_size):
        #   一维卷积，沿序列长度维度滑动卷积核
        #   in_channels=39: 39 种 MFCC 系数各为一个通道
        #   kernel_size=3: 每次看相邻 3 帧的局部窗口
        #   padding=1: 在序列两端各补一帧，使输出长度 = 输入长度（11 → 11）
        #   参数数量: in_channels * out_channels * kernel_size + bias
        #             = 39 * 128 * 3 + 128 = 15,104
        self.conv1 = nn.Conv1d(in_channels=39, out_channels=128, kernel_size=3, padding=1)
        # nn.BatchNorm1d: 批归一化（对每个输出通道做标准化）
        self.bn1 = nn.BatchNorm1d(128)

        # nn.Conv1d(128→256, k=3): 第二层卷积，输入通道=128（上一层的输出通道数）
        # 参数: 128 * 256 * 3 + 256 = 98,560
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        # nn.ReLU: 激活函数，f(x)=max(0,x)，引入非线性
        self.act_fn = nn.ReLU()

        # nn.AdaptiveAvgPool1d(1): 自适应均值池化，将任意长度的序列压缩到长度 1
        # 输入: (batch, 256, seq_len) → 输出: (batch, 256, 1)
        # 等价于对时间维度做全局平均，聚合 11 帧的信息为一帧的表示
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # nn.Dropout: 训练时以概率 p 随机丢弃神经元，防过拟合
        self.dropout = nn.Dropout(dropout)

        # === 分类头（全连接层）===
        # 池化后得到 256 维特征向量，通过两个全连接层映射到 39 类
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 39)

    def forward(self, x):
        """
        前向传播
        输入 x: (batch, 429)
        输出:   (batch, 39)  — 39 类别的 raw logits
        """
        # -------- 重塑 --------
        # x.view(x.size(0), 39, 11): 将 (batch, 429) 重塑为 (batch, 39, 11)
        #   batch 维度不变，429 = 39 通道 × 11 帧
        #   view() 是零拷贝的张量变形操作，不改变内存布局
        batch_size = x.size(0)
        x = x.view(batch_size, 39, 11)  # (batch, 39, 11)

        # -------- 卷积层 1 --------
        x = self.conv1(x)    # Conv1d: (batch, 39, 11) → (batch, 128, 11)
        x = self.bn1(x)      # BatchNorm: 沿通道维度归一化
        x = self.act_fn(x)   # ReLU: 非线性

        # -------- 卷积层 2 --------
        x = self.conv2(x)    # Conv1d: (batch, 128, 11) → (batch, 256, 11)
        x = self.bn2(x)      # BatchNorm
        x = self.act_fn(x)   # ReLU

        # -------- 全局池化 --------
        # AdaptiveAvgPool1d(1): 将 (batch, 256, 11) → (batch, 256, 1)
        x = self.global_pool(x)
        # .squeeze(-1): 去掉最后一维长度为 1 的维度，(batch, 256, 1) → (batch, 256)
        x = x.squeeze(-1)

        # -------- 分类头 --------
        x = self.fc1(x)      # Linear: (batch, 256) → (batch, 128)
        x = self.bn_fc(x)    # BatchNorm
        x = self.act_fn(x)   # ReLU
        x = self.dropout(x)  # Dropout

        x = self.out(x)      # Linear: (batch, 128) → (batch, 39), raw logits
        return x


# 实例化模型
model = CNNPhonemeClassifier(dropout=DROPOUT_RATE).to(device)

# ============================================================
# 4. 损失函数 & 优化器 & 学习率调度
# ============================================================
# nn.CrossEntropyLoss: 交叉熵损失，内部对 logits 做 softmax
criterion = nn.CrossEntropyLoss()

# torch.optim.Adam: Adam 优化器，自适应学习率
# weight_decay: L2 正则化
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# CosineAnnealingLR: 余弦退火学习率，从 LR 逐步衰减到 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ============================================================
# 5. 训练循环
# ============================================================
print("\n开始训练 CNN...")
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- 训练阶段 ---
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()       # 清空梯度
        logits = model(xb)          # 前向传播
        loss = criterion(logits, yb)  # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

        preds = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)
        train_loss_sum += loss.item() * yb.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # --- 验证阶段 ---
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

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

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT_DIR / "best_model_cnn.pt")

    print(f"Epoch [{epoch:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\nCNN 最佳验证准确率: {best_val_acc:.4f}")


# ============================================================
# 6. 测试集预测
# ============================================================
model.load_state_dict(torch.load(OUT_DIR / "best_model_cnn.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)

# 生成提交文件
submission_path = OUT_DIR / "submission_cnn.csv"
with open(submission_path, "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(all_preds):
        f.write(f"{i},{y}\n")

print(f"预测完成，已保存至 {submission_path}")
