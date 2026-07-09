"""
作业2 - 任务2-1：音素分类 —— 参考代码本地化 (reference_local.py)
来源：Colab HW02-1.ipynb 示例代码
Mac / MPS 适配 + 中文注释
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ============================================================
# 1. 设置随机种子，保证训练可复现
# ============================================================
# torch.backends.cudnn.deterministic: 强制使用确定性卷积算法
# 设为 True 后同一段代码每次运行结果完全一致，方便调试对比
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # 关闭自动寻找最优卷积算法（配合 deterministic 使用）

# 设置各层随机种子
myseed = 42069
np.random.seed(myseed)   # numpy 随机种子
torch.manual_seed(myseed)  # PyTorch CPU 随机种子

# ============================================================
# 2. 加载数据
# ============================================================
# __file__: 当前脚本的完整路径
# resolve(): 解析为绝对路径，消除 ../ 等相对符号
# parent.parent.parent: code/mlp/reference_local.py → code/mlp → code → 作业2
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/timit_11/timit_11"

# np.load: 加载 .npy 格式的 numpy 数组文件（保存时用的是 np.save）
train = np.load(DATA_DIR / "train_11.npy")                  # 训练特征，(1229932, 429)
train_label = np.load(DATA_DIR / "train_label_11.npy")      # 训练标签，(1229932,)，值域 0-38
test = np.load(DATA_DIR / "test_11.npy")                    # 测试特征，(451552, 429)

# astype(np.int64): 标签在 .npy 文件中存储为字符串（如 '0', '1'），转为 64 位整数
train_label = train_label.astype(np.int64)

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")


# ============================================================
# 3. 自定义 Dataset
# ============================================================
# Dataset: PyTorch 数据集的抽象基类
# 继承后需要实现: __init__(初始化), __getitem__(按索引取数据), __len__(返回总长度)
class TIMITDataset(Dataset):
    """TIMIT 音素分类数据集，封装特征和标签"""

    def __init__(self, X, y=None):
        # torch.from_numpy: numpy 数组 → PyTorch 张量（共享内存，不复制，速度快）
        # .float(): 转换为 32 位浮点数，神经网络运算需要 float32 类型
        self.data = torch.from_numpy(X).float()

        if y is not None:
            # torch.LongTensor: 整数 numpy 数组 → 64 位长整型张量
            # CrossEntropyLoss 要求标签为 LongTensor（整数类型）
            self.label = torch.LongTensor(y)
        else:
            self.label = None  # 测试集没有标签

    def __getitem__(self, idx):
        """DataLoader 调用此方法，按索引 idx 取出一条 (特征, 标签) 对"""
        if self.label is not None:
            return self.data[idx], self.label[idx]  # 训练/验证集：返回二元组
        return self.data[idx]                       # 测试集：只返回特征

    def __len__(self):
        """DataLoader 用它计算一个 epoch 内有多少个 batch"""
        return len(self.data)


# ============================================================
# 4. 划分训练集 / 验证集
# ============================================================
VAL_RATIO = 0.2   # 验证集占整个训练集的 20%（即 80% 训练 + 20% 验证）
percent = int(train.shape[0] * (1 - VAL_RATIO))  # 训练集样本数

# numpy 切片：[:percent] 前 80%，[percent:] 后 20%
train_x, train_y = train[:percent], train_label[:percent]
val_x, val_y = train[percent:], train_label[percent:]

print(f"训练集大小: {train_x.shape}  ({len(train_x):,} 样本)")
print(f"验证集大小: {val_x.shape}  ({len(val_x):,} 样本)")


# ============================================================
# 5. 构建 DataLoader（批量数据加载器）
# ============================================================
BATCH_SIZE = 64  # 每个 batch 喂 64 个样本（参考代码原值）

# DataLoader: 将 Dataset 包装成可迭代的 batch 加载器
# - shuffle=True: 每个 epoch 开始时重新打乱顺序，增加训练随机性（仅训练集）
# - shuffle=False: 验证/测试按顺序加载，保证结果可复现
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 6. 模型定义 —— 单独成员变量风格（参考代码原始写法）
# ============================================================
# 每层都单独声明为 self.xxx，需要手动在 forward 里逐行串联
# 优点：每层有独立名字，方便单独调试/访问某层权重
# 缺点：层数多时代码冗长
class Classifier(nn.Module):
    """4 层 MLP: 429 → 2048 → 1024 → 512 → 128 → 39"""

    def __init__(self):
        super(Classifier, self).__init__()

        # === 全连接层（Linear 层）===
        # nn.Linear(in, out): 执行 y = x @ W^T + b 的线性变换
        # 权重矩阵 W 的形状: (out_features, in_features) = (2048, 429)
        # 偏置向量 b 的形状: (out_features,) = (2048,)
        # 参考代码的隐藏层较宽（2048 起），模型容量比我们的 MLP 更大
        self.layer0 = nn.Linear(429, 2048)   # 第一层: 429 → 2048
        self.layer1 = nn.Linear(2048, 1024)  # 第二层: 2048 → 1024
        self.layer2 = nn.Linear(1024, 512)   # 第三层: 1024 → 512
        self.layer3 = nn.Linear(512, 128)    # 第四层: 512 → 128
        self.out = nn.Linear(128, 39)        # 输出层: 128 → 39（39 个音素类别）

        # === 激活函数 ===
        # nn.ReLU(): ReLU 激活函数 f(x) = max(0, x)
        # 没有可学习参数，所有层共用一个实例即可（不像 Linear 每层需要独立的 W 和 b）
        self.act_fn = nn.ReLU()

        # === Dropout ===
        # nn.Dropout(p): 训练时以概率 p 随机将神经元输出置零
        # p=0.25 表示每次前向传播有 25% 的神经元被"关掉"
        # 作用是防止模型过度依赖某些特定神经元，提升泛化能力
        self.dropout = nn.Dropout(0.25)

        # === 批归一化层（BatchNorm1d）===
        # nn.BatchNorm1d(num_features): 对每个特征维度独立做归一化
        # 公式: y = (x - E[x]) / sqrt(Var(x) + eps) * gamma + beta
        #   E[x] = 当前 batch 在该维度的均值（自动计算）
        #   Var(x) = 当前 batch 在该维度的方差（自动计算）
        #   gamma（缩放系数）和 beta（平移系数）是可学习参数
        # 作用: 缓解内部协变量偏移(Internal Covariate Shift)，让训练更稳定
        # 注意: num_features 必须等于对应 Linear 层的输出维度
        self.batchnorm0 = nn.BatchNorm1d(2048)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """前向传播: 输入 (batch, 429) → 输出 (batch, 39)"""
        x = self.layer0(x)      # Linear: 429 → 2048
        x = self.batchnorm0(x)   # BatchNorm: 归一化 2048 维
        x = self.act_fn(x)       # ReLU: 非线性激活
        x = self.dropout(x)      # Dropout: 随机丢弃 25%

        x = self.layer1(x)      # Linear: 2048 → 1024
        x = self.batchnorm1(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer2(x)      # Linear: 1024 → 512
        x = self.batchnorm2(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer3(x)      # Linear: 512 → 128
        x = self.batchnorm3(x)
        x = self.act_fn(x)

        # 输出层: 只做线性映射，不加 BN/ReLU/Dropout
        # CrossEntropyLoss 内部会自动做 softmax，我们需要 raw logits
        x = self.out(x)         # Linear: 128 → 39
        return x


# ============================================================
# 7. 设备选择
# ============================================================
def get_device():
    """自动选择可用设备: CUDA > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"            # Mac M1/M2/M3 等 Apple Silicon GPU
    return "cpu"

device = get_device()
print(f"\n使用设备: {device}")


# ============================================================
# 8. 训练配置
# ============================================================
num_epoch = 40              # 训练 40 轮（看 40 遍整个训练集）
learning_rate = 0.0001     # 学习率：每步参数更新的步长
model_path = "model.ckpt"  # 最优模型保存路径

# 实例化模型并移到指定设备
model = Classifier().to(device)

# nn.CrossEntropyLoss: 多分类交叉熵损失
# 输入: raw logits (batch, 39)，无需预先 softmax
# 目标: 整数类别索引 (batch,)，取值范围 0~38
criterion = nn.CrossEntropyLoss()

# torch.optim.NAdam: Nadam 优化器
# 结合了 Adam 的自适应学习率 + Nesterov 动量加速
# 比纯 Adam 收敛通常更稳定
# 注意: 原始参考代码写的是 NADM（笔误），正确名称是 NAdam
optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)


# ============================================================
# 9. 训练循环
# ============================================================
best_acc = 0.0

print("\n开始训练...")
for epoch in range(num_epoch):
    train_acc = 0.0    # 本 epoch 训练集累计正确数
    train_loss = 0.0   # 本 epoch 训练集累计损失
    val_acc = 0.0      # 本 epoch 验证集累计正确数
    val_loss = 0.0     # 本 epoch 验证集累计损失

    # --- 训练阶段 ---
    # model.train(): 切换为训练模式
    # 效果: BatchNorm 用当前 batch 的统计量; Dropout 启用随机丢弃
    model.train()

    # enumerate(train_loader): 遍历所有训练 batch
    # data 是 (inputs, labels) 二元组（由 TIMITDataset.__getitem__ 返回）
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # .to(device): 将张量移到指定设备（CPU/MPS）
        # Mac 上如果 device=mps，这里会把数据移到 Apple Silicon GPU

        # optimizer.zero_grad(): 清空上一轮计算的梯度
        # PyTorch 默认累积梯度（为了支持梯度累加），所以每轮要手动清零
        optimizer.zero_grad()

        # model(inputs): 前向传播
        # 输入 (batch, 429) → 输出 raw logits (batch, 39)
        outputs = model(inputs)

        # criterion(outputs, labels): 计算交叉熵损失
        # 返回值是标量（单个 Python float），表示当前 batch 的平均损失
        batch_loss = criterion(outputs, labels)

        # torch.max(outputs, 1): 沿维度 1（类别维度）取最大值
        # 返回值: (最大值 tensor, 最大值索引 tensor)
        # 索引就是模型预测的类别
        _, train_pred = torch.max(outputs, 1)

        # batch_loss.backward(): 反向传播
        # 自动应用链式法则，计算所有参数相对于 batch_loss 的梯度
        # 梯度结果存储在各参数的 .grad 属性中
        batch_loss.backward()

        # optimizer.step(): 根据梯度更新模型参数
        # 每个参数: param = param - lr * param.grad
        optimizer.step()

        # 累加统计量（用于 epoch 末计算平均值）
        # .item(): 单元素 tensor → Python 数值（避免 GPU↔CPU 传输开销）
        # train_pred.cpu() == labels.cpu(): 逐元素比较，返回 bool tensor
        # .sum().item(): 统计正确的样本数
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # --- 验证阶段 ---
    # model.eval(): 切换为评估模式
    # 效果: BatchNorm 用训练时累积的移动平均; Dropout 关闭（不做随机丢弃）
    model.eval()

    # torch.no_grad(): 关闭梯度计算
    # 推理阶段不需要反向传播，关闭后可减少显存占用、加速运算
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, val_pred = torch.max(outputs, 1)

            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += batch_loss.item()

    # 打印本 epoch 结果（每 epoch 打印一次）
    train_acc_ratio = train_acc / len(train_set)   # 训练准确率
    train_loss_avg = train_loss / len(train_loader)  # 训练平均损失
    val_acc_ratio = val_acc / len(val_set)         # 验证准确率
    val_loss_avg = val_loss / len(val_loader)      # 验证平均损失

    print(
        f"[{epoch + 1:03d}/{num_epoch:03d}] "
        f"Train Acc: {train_acc_ratio:.4f}  Loss: {train_loss_avg:.4f} | "
        f"Val Acc: {val_acc_ratio:.4f}  Loss: {val_loss_avg:.4f}"
    )

    # 如果当前验证准确率超过历史最佳，保存模型
    if val_acc > best_acc:
        best_acc = val_acc
        # torch.save: 将模型的所有可学习参数（权重、偏置）保存到文件
        # model.state_dict(): 返回参数字典 {参数名: tensor 值}
        torch.save(model.state_dict(), model_path)
        print(f"  ★ 保存最佳模型，验证 Acc: {best_acc / len(val_set):.4f}")

print(f"\n{'='*50}")
print(f"训练完成！最佳验证准确率: {best_acc / len(val_set):.4f}")
print(f"{'='*50}")


# ============================================================
# 10. 在测试集上预测
# ============================================================
# TIMITDataset(test, None): 测试集没有标签，传入 None
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 重新实例化模型并加载最优权重
model = Classifier().to(device)
# torch.load: 从 .ckpt 文件加载权重
# map_location="cpu": 即使模型是用 GPU 训练的，也能在 CPU/MPS 上加载
model.load_state_dict(torch.load(model_path, map_location="cpu"))

# model.eval(): 切换到评估模式（关闭 Dropout，BatchNorm 用移动平均）
model.eval()

predict = []
with torch.no_grad():
    for inputs in test_loader:
        # 测试集 DataLoader 返回只有特征（没有标签）
        inputs = inputs.to(device)
        outputs = model(inputs)

        # torch.max(outputs, 1): 取每个样本概率最大的类别
        _, test_pred = torch.max(outputs, 1)

        # .cpu().numpy(): 先从设备移回 CPU，再转 numpy 数组
        # .tolist(): 转为 Python 普通列表
        predict.extend(test_pred.cpu().numpy().tolist())

# 生成 Kaggle 提交文件: 两列 —— Id（行号）和 Class（预测类别 0-38）
with open("prediction.csv", "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(predict):
        f.write(f"{i},{y}\n")

print(f"预测完成，共 {len(predict):,} 条结果，已保存至 prediction.csv")
