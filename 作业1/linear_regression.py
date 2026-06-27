"""
线性回归 —— 新冠每日确诊病例预测
============================
模型说明：y = xW^T + b，即简单的线性加权求和
对比 MLP 可以看到"加非线性层"到底带来了多少提升
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
# 1. 设置随机种子（保证结果可复现）
# ============================================================
# torch.manual_seed：固定 PyTorch 的随机种子，让每次运行生成的随机数都一样
# 这样你多次跑代码得到的结果是一致的，方便调试
torch.manual_seed(42)

# ============================================================
# 2. 超参数（可以自己改着玩）
# ============================================================
BATCH_SIZE = 64          # 每次送入网络多少个样本一起算
LEARNING_RATE = 1e-3     # 学习率：每次参数更新的步长，太大容易震荡，太小收敛慢
EPOCHS = 300             # 整个训练集反复学多少遍
VALID_RATIO = 0.2        # 从训练集中拿出多少比例做验证集（20%）

# ============================================================
# 3. 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "covid.train.csv"
TEST_PATH = BASE_DIR / "covid.test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sampleSubmission.csv"
OUTPUT_PATH = BASE_DIR / "linear_submission.csv"

# ============================================================
# 4. 数据加载与预处理
# ============================================================
def load_data():
    """
    加载训练集和测试集，返回：
    X_train, y_train, X_val, y_val, X_test, scaler_mean, scaler_std
    """
    # pandas.read_csv：读取 CSV 文件为 DataFrame 表格
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # 训练集：去除 id 列（第 0 列），所有行，从第 1 列开始取
    # .values 把 pandas DataFrame 转为 numpy 数组，后面才能转成 torch 张量
    train_data = df_train.values  # shape: (2700, 95)
    test_data = df_test.values    # shape: (893, 94) —— 测试集没有目标值列

    # 特征（X）：所有行，从第 1 列到倒数第 2 列（即排除 id 和最后一列目标值）
    # 训练集最后一列是 tested_positive（目标值），前 1~93 列是特征
    X_all = train_data[:, 1:-1]   # shape: (2700, 93)，取第1列到倒数第2列
    # 标签（y）：最后一列 tested_positive，shape: (2700,)
    # reshape(-1, 1) 把一维数组变成二维列向量，方便和模型输出做运算
    y_all = train_data[:, -1].reshape(-1, 1)

    # 测试集特征：去除 id 列，剩下的全是特征（94-1=93 列）
    X_test = test_data[:, 1:]     # shape: (893, 93)

    # ---- 标准化（Normalization / Standardization）----
    # 为什么要标准化：不同特征的数值范围差异很大（如 0~1 vs 0~100），
    # 如果不标准化，数值大的特征会主导梯度更新，模型学不好
    # 方法：对每个特征，减去均值，除以标准差，得到均值为 0、标准差为 1 的数据

    # numpy.mean：计算均值，axis=0 表示按列（每个特征）计算
    # keepdims=True 保持维度，方便后面广播运算
    feat_mean = np.mean(X_all, axis=0, keepdims=True)   # shape: (1, 93)
    # numpy.std：计算标准差，ddof=0 表示总体标准差
    feat_std = np.std(X_all, axis=0, keepdims=True)     # shape: (1, 93)
    # 防止除以零：如果某个特征所有值都一样（标准差为 0），除 0 会出问题
    # 这里加一个极小值 1e-8 保护一下
    feat_std = np.clip(feat_std, a_min=1e-8, a_max=None)

    # 标准化公式：(x - mean) / std
    X_all_norm = (X_all - feat_mean) / feat_std   # shape: (2700, 93)
    X_test_norm = (X_test - feat_mean) / feat_std  # 用训练集的 mean/std 标准化测试集！

    # ---- 划分训练集和验证集 ----
    # 随机打乱索引，保证训练/验证分布一致
    # numpy.random.permutation：生成一个随机排列的索引数组
    n_train = X_all_norm.shape[0]                     # 2700
    indices = np.random.permutation(n_train)          # 打乱后的索引
    n_val = int(n_train * VALID_RATIO)                # 540 个验证样本

    val_idx = indices[:n_val]        # 前 540 个索引 → 验证集
    train_idx = indices[n_val:]      # 剩下的 2160 个 → 训练集

    X_train = X_all_norm[train_idx]    # shape: (2160, 93)
    y_train = y_all[train_idx]         # shape: (2160, 1)
    X_val = X_all_norm[val_idx]        # shape: (540, 93)
    y_val = y_all[val_idx]             # shape: (540, 1)

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"测试集: {X_test_norm.shape[0]} 个样本")
    print(f"特征维度: {X_train.shape[1]} 维")

    return X_train, y_train, X_val, y_val, X_test_norm


# numpy 数组 → torch 张量 → DataLoader
def make_loader(X, y, batch_size, shuffle=True):
    """
    将 numpy 数组包装成 PyTorch 的 DataLoader，方便分批训练

    DataLoader 的作用：自动把数据分成一个个小批次（batch），
    每个 epoch 按批次送给模型训练，不用自己写 for 循环切片
    """
    # torch.tensor：将 numpy 数组转为 PyTorch 张量
    # dtype=torch.float32：所有数据用 32 位浮点数，这是 PyTorch 默认精度
    # 为什么要转成 tensor？因为 PyTorch 的所有计算（加减乘除、梯度）都基于 tensor
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # TensorDataset：把 X 和 y 打包成一个"配对"的数据集对象
    # 每次取数据时 X 和 y 自动对应同一个样本
    dataset = TensorDataset(X_t, y_t)

    # DataLoader：在上面说的数据集基础上，加上分批、打乱等功能
    # shuffle=True：每个 epoch 开始时重新打乱数据，防止模型记住顺序
    # 这里去掉 drop_last，保证最后一个 batch 也能被处理
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# ============================================================
# 5. 定义线性回归模型
# ============================================================
class LinearRegression(nn.Module):
    """
    线性回归模型：y = xW^T + b

    nn.Module：所有 PyTorch 模型的基类，你的模型必须继承它
    它提供了参数管理、 .to(device)、 .train()/eval() 等基础设施
    """
    def __init__(self, input_dim):
        """
        input_dim：输入特征的维度（这里 = 93）
        """
        # super().__init__()：调用父类 nn.Module 的初始化方法
        # 必须写这行，否则模型会报错
        super().__init__()

        # nn.Linear：全连接层（线性层），做的事情就是 y = xW^T + b
        # 参数：
        #   in_features: 输入特征数（93）
        #   out_features: 输出特征数（1）——因为我们要预测一个数值
        # 实际上 nn.Linear 内部维护了权重 W（93×1 矩阵）和偏置 b（1 维标量）
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        """
        forward：前向传播函数
        当模型接收输入 x 时，自动调用这个函数计算结果
        PyTorch 会自动计算梯度，不需要你手动写反向传播
        """
        return self.linear(x)


# ============================================================
# 6. 训练函数
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    """
    训练模型并记录训练/验证的损失

    参数：
    - model: 要训练的模型
    - train_loader: 训练数据的 DataLoader
    - val_loader: 验证数据的 DataLoader
    - epochs: 训练轮数
    - lr: 学习率
    """
    # nn.MSELoss：均方误差损失函数，公式：loss = mean((y_pred - y_true)^2)
    # 对于回归问题（预测连续数值），MSE 是最常用的损失函数
    # 为什么用它？因为 MSE 对大误差惩罚更大（平方放大），促使模型更关注大误差样本
    criterion = nn.MSELoss()

    # optim.Adam：Adam 优化器，目前最常用的优化算法之一
    # 它会自动调整每个参数的学习率，比普通的 SGD（随机梯度下降）更好用
    # 参数 model.parameters()：告诉优化器要更新哪些参数（即模型的所有权重和偏置）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录每个 epoch 的损失，后面用来画图
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = BASE_DIR / "linear_best_model.pth"

    print("\n开始训练...")
    for epoch in range(epochs):
        # ---- 训练阶段 ----
        # model.train()：把模型切换到"训练模式"
        # 影响 Dropout、BatchNorm 等层的行为（虽然这里没有，但养成习惯）
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # 遍历 DataLoader 中的每个 batch
        # batch_x: 一批特征数据，shape = (batch_size, 93)
        # batch_y: 一批标签，shape = (batch_size, 1)
        for batch_x, batch_y in train_loader:
            # 前向传播：把数据送进模型，得到预测值
            predictions = model(batch_x)  # shape: (batch_size, 1)

            # 计算损失：预测值和真实值之间的差距
            loss = criterion(predictions, batch_y)

            # ---- 反向传播三部曲 ----
            # 1. optimizer.zero_grad()：清空上一步的梯度
            #    每次反向传播前必须调用，否则梯度会累加
            optimizer.zero_grad()
            # 2. loss.backward()：自动计算所有参数的梯度（链式法则）
            #    PyTorch 自动帮你算导数，不需要手动写求导公式
            loss.backward()
            # 3. optimizer.step()：根据梯度更新参数
            #    执行 W = W - lr * dW, b = b - lr * db
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # ---- 验证阶段 ----
        # model.eval()：切换到"评估模式"，禁用 dropout 等训练特有的操作
        model.eval()
        val_loss = 0.0
        val_batches = 0
        # torch.no_grad()：告诉 PyTorch 不要计算梯度
        # 验证时不需要反向传播，关闭梯度计算可以节省内存和加速
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # 保存验证损失最小的模型（早停：防止过拟合）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save：把模型参数保存到文件
            # 只保存 state_dict（参数）而不是整个模型，更轻量、更安全
            torch.save(model.state_dict(), best_model_path)

        # 每 50 轮打印一次进度
        if (epoch + 1) % 50 == 0:
            # numpy.sqrt：计算平方根，把 MSE 转为 RMSE
            # RMSE 的单位和原始数据一样，更直观
            train_rmse = np.sqrt(avg_train_loss)
            val_rmse = np.sqrt(avg_val_loss)
            print(f"Epoch {epoch+1:3d}/{epochs}  |  "
                  f"Train RMSE: {train_rmse:.4f}  |  "
                  f"Val RMSE: {val_rmse:.4f}")

    print(f"\n训练完成！最佳验证 RMSE: {np.sqrt(best_val_loss):.4f}")
    print(f"最佳模型已保存到: {best_model_path}")

    return train_losses, val_losses


# ============================================================
# 7. 预测函数
# ============================================================
def predict(model, X_test, output_path):
    """
    用训练好的模型预测测试集，并保存为提交文件
    """
    # 加载最佳模型参数
    # load_state_dict：把之前保存的参数恢复到模型中
    model.load_state_dict(torch.load(BASE_DIR / "linear_best_model.pth"))
    model.eval()

    # 转成 tensor
    X_t = torch.tensor(X_test, dtype=torch.float32)

    # torch.no_grad()：预测也不需要梯度
    with torch.no_grad():
        predictions = model(X_t)  # shape: (893, 1)

    # .numpy()：把 tensor 转回 numpy 数组
    # .flatten()：把二维 (893,1) 拉成一维 (893,)，方便存 CSV
    pred_values = predictions.numpy().flatten()

    # 加载 sampleSubmission.csv 模板，替换预测值
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    sample_sub["tested_positive"] = pred_values

    # 保存
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
    plt.savefig(BASE_DIR / "linear_loss_curve.png", dpi=150)
    print(f"损失曲线已保存")


# ============================================================
# 9. 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("线性回归 — 新冠每日确诊病例预测")
    print("=" * 50)

    # 加载数据
    X_train, y_train, X_val, y_val, X_test = load_data()

    # 创建 DataLoader
    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    # 创建模型
    # input_dim=93：93 个特征
    model = LinearRegression(input_dim=93)

    # 打印模型结构
    # print(model) 会调用模型的 __repr__ 方法，显示各层
    print(f"\n模型结构:\n{model}")

    # 统计模型参数量
    # sum(p.numel() for p in model.parameters())：计算所有参数的总个数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")  # 线性回归只有 93 + 1 = 94 个参数

    # 训练
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )

    # 预测测试集
    predict(model, X_test, OUTPUT_PATH)

    # 画损失曲线
    plot_losses(train_losses, val_losses, "Linear Regression - Loss Curve")

    print("\n✅ 线性回归完成！提交文件: linear_submission.csv")