"""
多层感知机（MLP）—— 新冠每日确诊病例预测
===============================
模型说明：在"线性层 + 激活函数 + 线性层 + 激活函数..."中堆叠，
引入非线性能力，拟合复杂关系

与 linear_regression.py 对比，核心区别只有"模型定义"部分。
数据预处理、训练流程完全一样，方便对比效果。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# 1. 随机种子
# ============================================================
torch.manual_seed(42)

# ============================================================
# 2. 超参数
# ============================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3          # Adam 通常 1e-3 是个好起点
EPOCHS = 500                   # MLP 参数多，多学几轮
VALID_RATIO = 0.2              # 验证集比例

# MLP 特有的超参数
HIDDEN1 = 128                  # 第一层隐藏层神经元个数
HIDDEN2 = 64                   # 第二层隐藏层神经元个数
DROPOUT = 0.2                  # Dropout 概率：训练时随机丢弃 20% 的神经元

# ============================================================
# 3. 路径
# ============================================================
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "covid.train.csv"
TEST_PATH = BASE_DIR / "covid.test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sampleSubmission.csv"
OUTPUT_PATH = BASE_DIR / "mlp_submission.csv"

# ============================================================
# 4. 数据加载与预处理（和线性回归完全一致）
# ============================================================
def load_data():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_data = df_train.values
    test_data = df_test.values

    X_all = train_data[:, 1:-1]
    y_all = train_data[:, -1].reshape(-1, 1)
    X_test = test_data[:, 1:]

    feat_mean = np.mean(X_all, axis=0, keepdims=True)
    feat_std = np.std(X_all, axis=0, keepdims=True)
    feat_std = np.clip(feat_std, a_min=1e-8, a_max=None)

    X_all_norm = (X_all - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    n_train = X_all_norm.shape[0]
    indices = np.random.permutation(n_train)
    n_val = int(n_train * VALID_RATIO)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X_all_norm[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all_norm[val_idx]
    y_val = y_all[val_idx]

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"测试集: {X_test_norm.shape[0]} 个样本")
    print(f"特征维度: {X_train.shape[1]} 维")

    return X_train, y_train, X_val, y_val, X_test_norm


def make_loader(X, y, batch_size, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# ============================================================
# 5. 定义 MLP 模型（核心区别在这里！）
# ============================================================
class MLPRegression(nn.Module):
    """
    多层感知机回归模型

    结构：Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear

    相比线性回归的区别：
    - 多个线性层堆叠（深度）
    - 每层之间插入激活函数（非线性能力）
    - 用 Dropout 防止过拟合
    """
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.2):
        """
        input_dim: 输入特征维度（93）
        hidden1:   第一层隐藏层神经元个数（128），越多模型容量越大
        hidden2:   第二层隐藏层神经元个数（64）
        dropout:   Dropout 概率（0.2），训练时随机丢弃 20% 的神经元
        """
        super().__init__()

        # nn.Sequential：一个"容器"，按顺序把各层排好
        # 数据输入后自动按顺序流过每一层，省去手动写 forward 里逐层调用的代码
        self.network = nn.Sequential(
            # ---- 第一层 ----
            # nn.Linear(93, 128)：将 93 维输入映射到 128 维
            # 为什么是 128？经验法则：隐藏层神经元数 = 输入维度的 1~2 倍
            nn.Linear(input_dim, hidden1),
            # nn.ReLU：激活函数，公式 f(x) = max(0, x)
            # 作用是引入非线性——如果没有它，多层线性叠加还是线性
            # 为什么用 ReLU？计算快，且能缓解梯度消失问题
            nn.ReLU(),
            # nn.Sigmoid
            # nn.Dropout：训练时随机把一些神经元的输出置为 0
            # 为什么用 Dropout？防止模型"死记硬背"训练数据（过拟合）
            # 随机丢弃迫使每个神经元都学到有用的特征，而不是依赖某些特定神经元
            nn.Dropout(dropout),

            # ---- 第二层 ----
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ---- 输出层 ----
            # nn.Linear(64, 1)：将 64 维隐层映射到 1 维输出（预测值）
            # 输出层不用激活函数——回归任务直接输出实数
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        # 数据依次流过 Sequential 中定义的每一层
        return self.network(x)


# ============================================================
# 6. 训练函数（和线性回归一样）
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = BASE_DIR / "mlp_best_model.pth"

    print("\n开始训练...")
    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 50 == 0:
            train_rmse = np.sqrt(avg_train_loss)
            val_rmse = np.sqrt(avg_val_loss)
            print(f"Epoch {epoch+1:3d}/{epochs}  |  "
                  f"Train RMSE: {train_rmse:.4f}  |  "
                  f"Val RMSE: {val_rmse:.4f}")

    print(f"\n训练完成！最佳验证 RMSE: {np.sqrt(best_val_loss):.4f}")
    print(f"最佳模型已保存到: {best_model_path}")

    return train_losses, val_losses


# ============================================================
# 7. 预测
# ============================================================
def predict(model, X_test, output_path):
    model.load_state_dict(torch.load(BASE_DIR / "mlp_best_model.pth"))
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(X_t)

    pred_values = predictions.numpy().flatten()

    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    sample_sub["tested_positive"] = pred_values
    sample_sub.to_csv(output_path, index=False)

    print(f"预测结果已保存到: {output_path}")
    print(f"预测值范围: {pred_values.min():.4f} ~ {pred_values.max():.4f}")

    return pred_values


# ============================================================
# 8. 画损失曲线
# ============================================================
def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.8)
    plt.plot(val_losses, label="Validation Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "mlp_loss_curve.png", dpi=150)
    print(f"损失曲线已保存")


# ============================================================
# 9. 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("MLP（多层感知机）— 新冠每日确诊病例预测")
    print("=" * 50)

    X_train, y_train, X_val, y_val, X_test = load_data()

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    # 创建 MLP 模型
    model = MLPRegression(
        input_dim=93,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        dropout=DROPOUT,
    )

    print(f"\n模型结构:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")
    # MLP 参数量 ≈ 93*128 + 128 + 128*64 + 64 + 64*1 + 1 ≈ 20289
    # 远远多于线性回归的 94 个参数

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )

    predict(model, X_test, OUTPUT_PATH)

    plot_losses(train_losses, val_losses, "MLP - Loss Curve")

    print("\n✅ MLP 完成！提交文件: mlp_submission.csv")