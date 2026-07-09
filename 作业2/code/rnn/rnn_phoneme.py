"""
作业2 - 任务2-1：音素分类 —— RNN（循环神经网络）版本
输入: 429 维 MFCC 特征（11帧 × 39维）
方法: 重塑为 (11, 39) 的时间序列，用 LSTM 建模帧与帧之间的时序依赖关系
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
#   标签是中心帧的音素类别
#   RNN 视角：429 → 重塑为 (11, 39)，视为以帧为时间步的序列
#   每个时间步 t 输入第 t 帧的 39 维 MFCC 特征
#   LSTM 会依次处理 t=1...11，利用隐藏状态在帧间传递信息
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
# 3. 模型定义 — RNN (LSTM)
# ============================================================
class RNNPhonemeClassifier(nn.Module):
    """
    RNN（LSTM）音素分类器

    输入是 429 维平铺向量（11帧 × 39维MFCC），
    RNN 的第一步将其重塑为 (batch, 11, 39)：
      - 11 = 序列长度（时间步 = 帧数）
      - 39 = 每步的特征维度（一帧的 MFCC 特征）

    用 LSTM 沿时间步依次处理，捕捉帧之间的长距离时序依赖。
    MLP 的局限是每帧独立预测；RNN 可以利用上下文信息。

    架构:
      输入 (batch, 429) → reshape → (batch, 11, 39)

      → LSTM(39 → 128, 2层, 双向)      # 建模时序依赖
        → 每帧输出的形状: (batch, 11, 256)  [双向128×2]
        → 对 11 帧做均值池化 → (batch, 256)

      → Linear(256→128) + BN + ReLU + Dropout    # 分类头
      → Linear(128→39)                            # 输出
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # nn.LSTM(input_size, hidden_size, num_layers, bidirectional, batch_first):
        #   input_size=39: 每个时间步输入的特征维度（一帧的 MFCC 维数）
        #   hidden_size=128: 隐藏状态 h 的维度
        #   num_layers=2: 堆叠 2 层 LSTM，增加模型深度
        #   bidirectional=True: 双向 LSTM，正向看过去→未来，反向看未来→过去
        #     这样每个时间步的表示同时包含前后文信息
        #     hidden_size * 2 = 256（双向拼接后）
        #   batch_first=True: 输入形状改为 (batch, seq_len, input_size)
        #     默认是 (seq_len, batch, input_size)
        #   参数数量 ≈ 4 × (input_size × hidden_size + hidden_size² + hidden_size)
        #   第1层: 4×(39×128 + 128² + 128) = 4×(4992+16384+128) = 86,016
        #   第2层: 4×(256×128 + 128² + 128) = 4×(32768+16384+128) = 197,120
        #   LSTM 总参数: 283,136
        # nn.Dropout(dropout): 除最后一层外，每层 LSTM 输出后加 dropout
        #   注意: LSTM 的 dropout 参数只在 num_layers>1 时有效
        self.lstm = nn.LSTM(
            input_size=39,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,  # 仅对多层 LSTM 的非最后层生效
        )

        # === 分类头 ===
        # LSTM 双向输出维度 = hidden_size * 2 = 256
        self.fc1 = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(128, 39)

    def forward(self, x):
        """
        前向传播
        输入 x: (batch, 429)
        输出:   (batch, 39)  — 39 类别的 raw logits
        """
        batch_size = x.size(0)

        # -------- 重塑 --------
        # x.view(batch, 11, 39): 429 维向量 → (11 帧, 39 维/帧) 的序列
        # 第 t 个时间步包含第 t 帧的 MFCC 特征
        x = x.view(batch_size, 11, 39)  # (batch, 11, 39)

        # -------- LSTM --------
        # self.lstm(x) 返回 (output, (h_n, c_n))
        #   output: 所有时间步的输出 (batch, 11, 256)
        #   h_n: 最后一层的隐藏状态 (num_layers * num_directions, batch, 128)
        #   c_n: 最后一层的细胞状态，形状同 h_n
        # 这里用 output（每帧的输出）做均值池化
        lstm_out, _ = self.lstm(x)  # (batch, 11, 256)

        # -------- 时序池化 --------
        # lstm_out.mean(dim=1): 对 11 个时间步的输出取均值
        # (batch, 11, 256) → (batch, 256)
        # 均值池化将 11 帧的信息聚合为一个固定长度的向量
        # 注意：中心帧对应 t=5（0-indexed），但池化利用了全部上下文
        x = lstm_out.mean(dim=1)

        # -------- 分类头 --------
        x = self.fc1(x)      # Linear: (batch, 256) → (batch, 128)
        x = self.bn(x)       # BatchNorm
        x = self.act_fn(x)   # ReLU
        x = self.dropout(x)  # Dropout
        x = self.out(x)      # Linear: (batch, 128) → (batch, 39)

        return x

    # 对于不想用均值池化、想用最后帧输出（或拼接首尾）的情况，可以改成:
    #
    # 方案 1: 拼接双向 LSTM 的首尾输出
    #   _, (h_n, _) = self.lstm(x)
    #   # h_n 形状: (4, batch, 128) [2层 × 2方向]
    #   # 取最后一层的正向和反向
    #   h_forward = h_n[-2]  # (batch, 128) — 正向最后帧
    #   h_backward = h_n[-1] # (batch, 128) — 反向最后帧（即序列第一帧）
    #   x = torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)
    #
    # 均值池化通常比单取最后帧更稳定，因为它利用了所有帧的信息


# 实例化模型
model = RNNPhonemeClassifier(dropout=DROPOUT_RATE).to(device)

# ============================================================
# 4. 损失函数 & 优化器 & 学习率调度
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ============================================================
# 5. 训练循环
# ============================================================
print("\n开始训练 RNN (LSTM)...")
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- 训练阶段 ---
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_: 梯度裁剪
        # 对 RNN 来说梯度容易爆炸（长序列上梯度连乘），裁剪到最大范数 max_norm 防止不稳定
        # 原理: 若 ||g|| > max_norm，则 g = g * max_norm / ||g||
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

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
        torch.save(model.state_dict(), OUT_DIR / "best_model_rnn.pt")

    print(f"Epoch [{epoch:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\nRNN 最佳验证准确率: {best_val_acc:.4f}")


# ============================================================
# 6. 测试集预测
# ============================================================
model.load_state_dict(torch.load(OUT_DIR / "best_model_rnn.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)

# 生成提交文件
submission_path = OUT_DIR / "submission_rnn.csv"
with open(submission_path, "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(all_preds):
        f.write(f"{i},{y}\n")

print(f"预测完成，已保存至 {submission_path}")
