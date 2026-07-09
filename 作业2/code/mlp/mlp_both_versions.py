"""
作业2 - 任务2-1：音素分类（ModuleList 版本）
用 ModuleList + for 循环的方式构建 MLP
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/timit_11/timit_11"
OUT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 256
EPOCHS = 12
SEED = 42
VALID_RATIO = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 加载数据 ====================
train_x = np.load(DATA_DIR / "train_11.npy").astype(np.float32)
train_y = np.load(DATA_DIR / "train_label_11.npy")
test_x = np.load(DATA_DIR / "test_11.npy").astype(np.float32)
train_y = train_y.astype(np.int64)

num_train = int(len(train_x) * (1 - VALID_RATIO))

# ==================== 构建 DataLoader ====================
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

print(f"训练集: {len(train_dataset)}  验证集: {len(valid_dataset)}  测试集: {len(test_x)}")


# ==================== 模型定义（ModuleList 版本） ====================
class MLPClassifier_ModuleList(nn.Module):
    """
    用 ModuleList + for 循环构建的 MLP 分类器

    结构: 429 → 512 → 256 → 128 → 39
    每层内部: Linear → BatchNorm1d → ReLU → Dropout
    """

    def __init__(self, input_dim=429, hidden_dims=[512, 256, 128], num_classes=39, dropout=0.3):
        super().__init__()

        # ---------- 第一步：构建层列表 ----------
        # ModuleList: 存储子模块的容器，等价于增强版 list
        # 作用：让 PyTorch 能通过 model.parameters() 找到这些层的权重
        self.layers = nn.ModuleList()

        # dims 是"维度接缝点"列表
        # input_dim=429, hidden_dims=[512, 256, 128]
        # dims = [429, 512, 256, 128]
        # 含义：[第0层输入, 第1层输入, 第2层输入, 第3层输入]
        dims = [input_dim] + hidden_dims

        # 初始化时执行一次，动态生成所有隐藏层
        for i in range(len(dims) - 1):
            # nn.Linear: 全连接层 y = xW^T + b
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # nn.BatchNorm1d: 批归一化，加速收敛
            self.layers.append(nn.BatchNorm1d(dims[i + 1]))

            # nn.ReLU: 激活函数 f(x)=max(0,x)，引入非线性
            self.layers.append(nn.ReLU())

            # nn.Dropout: 随机丢弃神经元，防止过拟合
            self.layers.append(nn.Dropout(dropout))

        # 输出层单独声明，不在循环中
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        # 推理时执行：将数据依次流过每一层
        # 这是每次前向传播都要跑的循环
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x


# ==================== 模型定义（Sequential 版本） ====================
class MLPClassifier_Sequential(nn.Module):
    """
    用 nn.Sequential 构建的 MLP 分类器

    结构和 ModuleList 版本完全相同，只是写法不同
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # nn.Sequential: 自动按顺序串联各层的容器
        # 调用 self.network(x) 时会自动依次流过其中定义的每一层
        self.network = nn.Sequential(
            # ===== 第一层: 429 → 512 =====
            nn.Linear(429, 512),       # 全连接层
            nn.BatchNorm1d(512),       # 批归一化
            nn.ReLU(),                 # 激活函数
            nn.Dropout(dropout),       # 随机丢弃

            # ===== 第二层: 512 → 256 =====
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ===== 第三层: 256 → 128 =====
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ===== 输出层: 128 → 39 =====
            # 不加 BN/激活，直接输出 raw logits
            nn.Linear(128, 39),
        )

    def forward(self, x):
        # Sequential 自动处理所有层的前向传播
        return self.network(x)


# ==================== 验证两个模型等价 ====================
# 用相同的随机输入检查两个模型的输出和参数量是否一致
torch.manual_seed(0)
model_a = MLPClassifier_ModuleList()
torch.manual_seed(0)
model_b = MLPClassifier_Sequential()

dummy_input = torch.randn(4, 429)
out_a = model_a(dummy_input)
out_b = model_b(dummy_input)

print(f"\n=== 等价性验证 ===")
# 两个模型参数数量对比
params_a = sum(p.numel() for p in model_a.parameters())
params_b = sum(p.numel() for p in model_b.parameters())
print(f"ModuleList 版本参数量: {params_a:,}")
print(f"Sequential 版本参数量: {params_b:,}")
print(f"参数量一致: {params_a == params_b}")

# 前向输出对比（ModuleList 的 classifier 权重和 Sequential 最后一层 Linear 权重对应）
# 注意：因为两个模型各自独立初始化，权重不同，所以输出不会相等
# 但网络拓扑和参数量完全相同
print(f"\nModuleList forward 输出形状: {out_a.shape}")
print(f"Sequential forward 输出形状: {out_b.shape}")


# ==================== 训练（用 Sequential 版本） ====================
model = MLPClassifier_Sequential(dropout=DROPOUT_RATE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n开始训练...")
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)
        train_loss_sum += loss.item() * yb.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

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
        torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

    print(
        f"Epoch [{epoch:2d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

print(f"\n最佳验证准确率: {best_val_acc:.4f}")

# ==================== 预测 & 提交 ====================
model.load_state_dict(torch.load(OUT_DIR / "best_model.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)

import csv
submission_path = OUT_DIR / "submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Class"])
    for idx, pred in enumerate(all_preds):
        writer.writerow([idx, pred])

print(f"提交文件已保存: {submission_path}")
