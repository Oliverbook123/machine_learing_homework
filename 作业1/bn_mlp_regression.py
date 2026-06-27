"""
带有 Batch Normalization（批归一化）的 MLP 回归模型
================================================
与 mlp_regression.py 的区别：
- 在每个 Linear 和 ReLU 之间插入 nn.BatchNorm1d 层
- 输出层之前不放 BN（回归任务，输出是单个实数，不需要归一化）

Batch Normalization 的作用（与普通特征归一化的区别）：
- 特征归一化（Z-score）：训练开始前对原始输入 X 做一次归一化（代码中仍在做）
- BatchNorm：在神经网络**每一层内部**、激活函数之前，对当前 mini-batch 的输出做归一化
- BN 的好处：
  1. 加速收敛（可以用更大的学习率）
  2. 缓解梯度消失/爆炸
  3. 有一定正则化效果（mini-batch 的噪声类似注入随机性）
  4. 减少对权重初始化的敏感性
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
# 1. 随机种子 —— torch.manual_seed: 固定 PyTorch 的随机数，使每次运行结果可复现
# ============================================================
torch.manual_seed(42)

# ============================================================
# 2. 超参数
# ============================================================
BATCH_SIZE = 64
# 使用 BN 后收敛更快，可以用更大的学习率
LEARNING_RATE = 3e-3  # BN 允许更大学习率（普通 MLP 用 1e-3，这里提至 3e-3）
EPOCHS = 300           # BN 加速收敛，可以适当减少 epoch
VALID_RATIO = 0.2

# MLP 结构参数
HIDDEN1 = 128
HIDDEN2 = 64
DROPOUT = 0.2

# ============================================================
# 3. 路径
# ============================================================
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "covid.train.csv"
TEST_PATH = BASE_DIR / "covid.test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sampleSubmission.csv"
OUTPUT_PATH = BASE_DIR / "bn_mlp_submission.csv"


# ============================================================
# 4. 数据加载与预处理（和原始 MLP 一致：特征级 Z-score 归一化仍然要做）
# ============================================================
def load_data():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_data = df_train.values
    test_data = df_test.values

    # 第一列是 id，最后一列是目标值，中间是特征
    X_all = train_data[:, 1:-1]       # shape: (N, 93)
    y_all = train_data[:, -1].reshape(-1, 1)  # shape: (N, 1)
    X_test = test_data[:, 1:]          # shape: (N_test, 93)

    # ---- 特征级 Z-score 归一化：这一步仍然必须做 ----
    # np.mean(axis=0): 沿行（样本）方向求均值，得到每个特征的均值
    feat_mean = np.mean(X_all, axis=0, keepdims=True)   # shape: (1, 93)
    # np.std(axis=0): 沿行方向求标准差
    feat_std = np.std(X_all, axis=0, keepdims=True)     # shape: (1, 93)
    # np.clip: 把标准差的下限限制在 1e-8，防止除以 0
    feat_std = np.clip(feat_std, a_min=1e-8, a_max=None)

    # Z-score 标准化: (x - μ) / σ
    X_all_norm = (X_all - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std       # 测试集用训练集的 μ 和 σ

    # 划分训练集/验证集
    # np.random.permutation: 随机打乱索引，用于划分训练/验证集
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
    # torch.tensor: 将 numpy 数组转为 PyTorch 张量，dtype=torch.float32 指定 32 位浮点
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    # TensorDataset: 将特征和标签打包成一个数据集对象，DataLoader 可直接读取
    dataset = TensorDataset(X_t, y_t)
    # DataLoader: 按 batch_size 切分数据，shuffle=True 时每轮随机打乱（训练用）
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# ============================================================
# 5. 定义带 BatchNorm 的 MLP 模型（核心区别！）
# ============================================================
class BnMLPRegression(nn.Module):
    """
    带有 Batch Normalization 的多层感知机回归模型

    结构：Linear → BatchNorm1d → ReLU → Dropout
           → Linear → BatchNorm1d → ReLU → Dropout
           → Linear（输出层，不加 BN）

    与普通 MLP 的关键区别：
    - 每一层 Linear 输出后、ReLU 激活前，插入 BatchNorm1d
    - BatchNorm1d 会对当前 mini-batch 的该层输出做归一化：
      x_norm = (x - μ_batch) / σ_batch * γ + β
      其中 γ（weight）和 β（bias）是可学习参数，μ_batch 和 σ_batch 从当前 batch 估计
    - 输出层不加 BN：回归任务输出是 1 维实数，归一化会破坏预测值的尺度
    """

    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.2):
        """
        input_dim: 输入特征维度（93）
        hidden1:   第一层隐藏层神经元个数（128）
        hidden2:   第二层隐藏层神经元个数（64）
        dropout:   Dropout 概率（0.2）
        """
        super().__init__()

        # nn.Sequential: 有序容器，输入按顺序流过每一层
        self.network = nn.Sequential(
            # ========= 第一层 =========
            # nn.Linear(93 → 128): 全连接层，将 93 维输入线性映射到 128 维
            nn.Linear(input_dim, hidden1),
            # nn.BatchNorm1d(128): 批归一化 —— 对 128 维输出的每个维度，
            # 在当前 mini-batch 内做归一化：(x - μ_batch) / σ_batch * γ + β
            # γ 和 β 是可学习参数，让模型自主恢复原始表达能力
            nn.BatchNorm1d(hidden1),
            # nn.ReLU: 激活函数 f(x)=max(0,x)，引入非线性
            nn.ReLU(),
            # nn.Dropout(0.2): 训练时随机丢弃 20% 的神经元，防止过拟合
            nn.Dropout(dropout),

            # ========= 第二层 =========
            # nn.Linear(128 → 64): 将 128 维隐层输出映射到 64 维
            nn.Linear(hidden1, hidden2),
            # nn.BatchNorm1d(64): 对 64 维输出做批归一化
            nn.BatchNorm1d(hidden2),
            # nn.ReLU: 非线性激活
            nn.ReLU(),
            # nn.Dropout(0.2): 随机丢弃防止过拟合
            nn.Dropout(dropout),

            # ========= 输出层 =========
            # nn.Linear(64 → 1): 将 64 维映射到 1 维预测值
            # 输出层不加 BatchNorm：回归任务需要直接输出实数，归一化会破坏数值范围
            # 输出层不加 ReLU：ReLU 会强行把输出 ≥0，而预测值可能是任意实数
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        # 数据依次流过 Sequential 中定义的每一层
        return self.network(x)


# ============================================================
# 6. 训练函数（和原始 MLP 基本一致）
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    # nn.MSELoss: 均方误差损失，计算 (预测值 - 真实值)² 的均值
    criterion = nn.MSELoss()
    # optim.Adam: Adam 优化器，自适应学习率
    # model.parameters(): 返回模型中所有可训练参数（包括 BatchNorm 的 γ 和 β）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = BASE_DIR / "bn_mlp_best_model.pth"

    print("\n开始训练（BatchNorm MLP）...")
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        # model.train(): 切换到训练模式
        # 这对 BatchNorm 至关重要：train 模式下 BN 用当前 batch 的统计量；
        # eval 模式下 BN 用训练阶段累积的全局统计量（running_mean/running_var）
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # 前向传播：数据流过 Sequential → Linear → BN → ReLU → Dropout → ...
            predictions = model(batch_x)
            # 计算损失
            loss = criterion(predictions, batch_y)

            # optimizer.zero_grad(): 清空上一轮的梯度，防止梯度累加
            optimizer.zero_grad()
            # loss.backward(): 反向传播，计算所有参数的梯度
            loss.backward()
            # optimizer.step(): 根据梯度更新参数
            optimizer.step()

            # loss.item(): 将标量张量转为 Python float
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # ========== 验证阶段 ==========
        # model.eval(): 切换到评估模式
        # BatchNorm 关键：eval 模式下 BN 使用 running_mean/running_var（训练时累积的全局统计量）
        # 而不是当前 batch 的统计量，保证推理时的稳定性
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():  # 禁用梯度计算，节省显存/内存
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict()): 保存模型权重字典到文件
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 30 == 0:  # BN 收敛快，30 轮打印一次
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
    # model.load_state_dict(torch.load(...)): 加载训练时保存的最佳权重
    model.load_state_dict(torch.load(BASE_DIR / "bn_mlp_best_model.pth"))
    # model.eval(): 切换到评估模式，BN 使用全局统计量
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(X_t)

    # .numpy(): 将 PyTorch 张量转为 numpy 数组
    # .flatten(): 将多维数组展平为一维
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
    plt.savefig(BASE_DIR / "bn_mlp_loss_curve.png", dpi=150)
    print(f"损失曲线已保存")


# ============================================================
# 9. 同时运行原始 MLP 作为对照
# ============================================================
def run_baseline_mlp(X_train, y_train, X_val, y_val, X_test_norm):
    """运行原始 MLP（不加 BN）作为对比基准"""
    from mlp_regression import MLPRegression

    print("\n" + "=" * 50)
    print("【对照组】原始 MLP（无 BatchNorm）")
    print("=" * 50)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = MLPRegression(
        input_dim=93,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        dropout=DROPOUT,
    )

    print(f"\n模型结构:\n{model}")

    # 使用本文件的 train_model 训练（参数和 BN 版本一致，方便对比）
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    # 使用本文件的 predict 预测
    predict(model, X_test_norm, BASE_DIR / "mlp_baseline_submission.csv")
    return model


# ============================================================
# 10. 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BatchNorm MLP — 新冠每日确诊病例预测")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test_norm = load_data()

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    # 创建带 BatchNorm 的 MLP 模型
    bn_model = BnMLPRegression(
        input_dim=93,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        dropout=DROPOUT,
    )

    print(f"\n模型结构:\n{bn_model}")

    total_params = sum(p.numel() for p in bn_model.parameters())
    print(f"模型总参数量: {total_params}")
    # BatchNorm 额外增加了可学习参数：
    # 原始 MLP: ~20,289 个参数
    # BN MLP:   ~20,673 个参数（多了 BatchNorm1d 的 weight(γ) 和 bias(β)）
    #         第一层 BN 多了 128*2=256, 第二层 BN 多了 64*2=128，共多 384 个

    # 训练 BN 版本
    train_losses, val_losses = train_model(
        bn_model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )

    # 预测
    predict(bn_model, X_test_norm, OUTPUT_PATH)

    # 画损失曲线
    plot_losses(train_losses, val_losses, "BatchNorm MLP - Loss Curve")

    # 同时运行原始 MLP 做对比
    baseline = run_baseline_mlp(X_train, y_train, X_val, y_val, X_test_norm)

    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("  BN MLP 提交文件:    bn_mlp_submission.csv")
    print("  原始 MLP 对照文件:   mlp_baseline_submission.csv")
    print("=" * 60)