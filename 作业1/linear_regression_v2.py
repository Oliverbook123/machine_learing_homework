"""
线性回归 v2 —— 和 v1 的唯一区别：验证集划分方式
================================================
v1：np.random.permutation 随机打乱，80%/20%
v2：i % 10 == 0 的方式，90%/10%，确定性划分

其他一切完全一致，方便你对比两种划分方式的差异
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 300

# ============================================================
# 路径
# ============================================================
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "covid.train.csv"
TEST_PATH = BASE_DIR / "covid.test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sampleSubmission.csv"
OUTPUT_PATH = BASE_DIR / "linear_v2_submission.csv"


def load_data():
    """
    数据加载 + 预处理。
    ★ 关键区别：用 i % 10 == 0 划分验证集，而不是随机打乱
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_data = df_train.values
    test_data = df_test.values

    X_all = train_data[:, 1:-1]   # 特征: (2700, 93)
    y_all = train_data[:, -1].reshape(-1, 1)  # 标签: (2700, 1)
    X_test = test_data[:, 1:]     # 测试集特征: (893, 93)

    # ---- 标准化 ----
    feat_mean = np.mean(X_all, axis=0, keepdims=True)
    feat_std = np.std(X_all, axis=0, keepdims=True)
    feat_std = np.clip(feat_std, a_min=1e-8, a_max=None)

    X_all_norm = (X_all - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    # ★★★ 核心改动：按 i % 10 == 0 划分验证集 ★★★
    # 这种做法在时间序列数据中更合理，因为它保留了原始顺序信息
    # 每 10 条取 1 条作为验证集，剩下 9 条做训练
    # 这样验证集 = 270 个样本，训练集 = 2430 个样本
    val_idx = [i for i in range(X_all_norm.shape[0]) if i % 10 == 0]
    train_idx = [i for i in range(X_all_norm.shape[0]) if i % 10 != 0]

    X_train = X_all_norm[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all_norm[val_idx]
    y_val = y_all[val_idx]

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"测试集: {X_test_norm.shape[0]} 个样本")
    print(f"划分方式: i % 10 == 0 取验证集（确定性划分）")
    print(f"特征维度: {X_train.shape[1]} 维")

    return X_train, y_train, X_val, y_val, X_test_norm


def make_loader(X, y, batch_size, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = BASE_DIR / "linear_best_v2.pth"

    print("\n开始训练...")
    for epoch in range(epochs):
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


def predict(model, X_test, output_path):
    model.load_state_dict(torch.load(BASE_DIR / "linear_best_v2.pth"))
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
    plt.savefig(BASE_DIR / "linear_v2_loss_curve.png", dpi=150)
    print("损失曲线已保存")


if __name__ == "__main__":
    print("=" * 50)
    print("线性回归 v2（i % 10 划分验证集）")
    print("=" * 50)

    X_train, y_train, X_val, y_val, X_test = load_data()

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = LinearRegression(input_dim=93)
    print(f"\n模型结构:\n{model}")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())}")

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )

    predict(model, X_test, OUTPUT_PATH)
    plot_losses(train_losses, val_losses, "Linear Regression v2 (i%10 split)")

    print("\n✅ v2 完成！提交文件: linear_v2_submission.csv")