"""
作业2 - 任务2-2：黑塞矩阵 (Hessian Matrix)
判断训练后的模型处于"局部极小值类点"、"鞍点"还是"以上皆非"

原理：
  梯度为零时，通过黑塞矩阵（二阶导数矩阵）的特征值正负判断地形：
  - 特征值全部为正 → 山谷底（局部极小值）
  - 特征值有正有负 → 马鞍点
  - 梯度不为零 → 以上皆非

注意事项：
  1. 填写你的学号 student_id
  2. 需要网络环境来下载 checkpoint 文件
  3. 不需要修改其他代码
"""
import re
import warnings
import numpy as np
from math import pi
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ============================================================
# 0. 填写学号（！！！必须修改！！！）
# ============================================================
# 学号用于确定加载哪一个 checkpoint（取学号最后一位数字）
# 不同学号得到的模型权重不同，结果也不同
student_id = 'your_student_id'  # ← 在这里填写你的学号（如 'b06901020'）

assert student_id != 'your_student_id', '请先填写你的学号 student_id！'


# ============================================================
# 1. 下载依赖包和 checkpoint 数据
# ============================================================
# subprocess: 执行 shell 命令安装包和下载文件
import subprocess, sys, os

# 安装 autograd-lib: 用于计算 Hessian 矩阵的第三方库
# 原理：通过注册前向/反向钩子(hook)，自动收集每层的激活值和反向梯度来计算 Hessian
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'autograd-lib', '-q'])

# 安装 gdown: 用于从 Google Drive 下载文件的工具
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'])

# 下载助教提供的 checkpoint 文件 (data.pth)
# Google Drive 文件 ID: 1ym6G7KKNkbsqSnMmnxdQKHO1JBoF0LPR
# 这个文件包含 10 组（学号最后一位 0-9）预训练模型权重和对应训练数据
data_path = 'data.pth'
if not os.path.exists(data_path):
    print("正在下载 checkpoint 文件...")
    subprocess.check_call(['gdown', '--id', '1ym6G7KKNkbsqSnMmnxdQKHO1JBoF0LPR'])
    print("下载完成。")
else:
    print("checkpoint 文件已存在，跳过下载。")


# ============================================================
# 2. 定义目标函数模型
# ============================================================
# 助教训练了一个简单的 2 层 MLP 来拟合数学函数:
#   f(x) = sin(5πx) / (5πx)   (sinc 函数的变体)
# 这是一个单变量函数，所以输入输出都是 1 维
class MathRegressor(nn.Module):
    """
    数学函数拟合器：拟合 f(x) = sin(5πx) / (5πx)

    nn.Sequential: 层顺序容器，输入按顺序流过各层
      Linear(1→128): 输入 x（标量）映射到 128 维隐藏空间
      ReLU:          非线性激活
      Linear(128→1): 从隐藏空间映射回标量输出
    """
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),   # 输入层: 1 → 128
            nn.ReLU(),                  # 激活函数
            nn.Linear(num_hidden, 1)    # 输出层: 128 → 1
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


# ============================================================
# 3. 加载预训练模型和数据
# ============================================================
# 根据学号最后一位确定使用哪一组 checkpoint
# - 如果是数字: 直接取最后一位 (如 'b06901020' → key=0)
# - 如果是字母: 取 ASCII 码的个位数 (如 'b06901A20' → ord('A')=65 → key=5)
key = student_id[-1]
if re.match('[0-9]', key) is not None:
    key = int(key)
else:
    key = ord(key) % 10

print(f"学号: {student_id} → 使用 key={key} 的 checkpoint")

# data.pth 是一个字典:
#   data[key]['model']: 对应学号的预训练模型权重
#   data[key]['data']:  (train, target) 一批用于评估的训练数据
model = MathRegressor()

# autograd_lib.register(model): 注册 autograd-lib 的钩子
# 作用: 在 model 的每个子模块上安装 hook，使得可以捕获前向/反向的中间值
# 后续可以通过 autograd_lib.module_hook() 上下文管理器来激活这些钩子
from autograd_lib import autograd_lib
autograd_lib.register(model)

# torch.load: 加载 .pth 文件（PyTorch 的序列化格式）
# map_location='cpu': 即使 model 是 GPU 训练的，也在 CPU 上加载
data = torch.load(data_path, map_location='cpu')[key]
model.load_state_dict(data['model'])
train, target = data['data']

print(f"训练数据形状: {train.shape}")   # 一批训练样本
print(f"目标值形状: {target.shape}")


# ============================================================
# 4. 计算梯度范数 (Gradient Norm)
# ============================================================
def compute_gradient_norm(model, criterion, train, target):
    """
    计算模型参数的梯度范数（一阶导数的大小）

    梯度范数衡量了"当前点有多陡峭"：
    - 梯度范数 ≈ 0 → 处于平坦区域（可能是极小值或鞍点）
    - 梯度范数 > 0 → 还有下降空间

    计算步骤:
    1. 前向传播: 计算模型输出和损失
    2. loss.backward(): 反向传播，自动计算各参数梯度
    3. 收集各 Linear 层的权重梯度，计算其 L2 范数（欧几里得长度）
    4. 对所有层的范数取均值
    """
    model.train()
    model.zero_grad()                 # 清空之前可能的梯度

    output = model(train)              # 前向传播: 输入 → 模型 → 输出
    loss = criterion(output, target)   # 计算 MSE 损失
    loss.backward()                    # 反向传播: 自动计算所有参数的梯度

    grads = []
    # 遍历模型的每一层，只处理 Linear 层（有可学习权重的层）
    for p in model.regressor.children():
        if isinstance(p, nn.Linear):
            # p.weight.grad: 权重的梯度（与权重形状相同）
            # .norm(2): 计算 L2 范数（各元素平方和的开方）
            # .item(): 从 torch 标量转为 Python float
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)

    grad_mean = np.mean(grads)  # 各层梯度范数的均值
    return grad_mean


# ============================================================
# 5. 计算极小值比例 (Minimum Ratio)
# ============================================================
# 黑塞矩阵 H 是损失函数对模型参数的二阶偏导数矩阵
# H 的特征值正负决定了当前位置的地形性质:
#   特征值全部 > 0 → 局部极小值（各方向都向上弯曲）
#   特征值全部 < 0 → 局部极大值（各方向都向下弯曲）
#   特征值有正有负 → 鞍点（某些方向向上，某些向下）

# 用 autograd-lib 计算黑塞矩阵的近似值
# 原理: https://en.wikipedia.org/wiki/Gauss–Newton_algorithm
# 近似 Hessian = (B × A) × (B × A)^T
# 其中 A = 层输入, B = 反向传播值（梯度）

# 存储每一层的输入激活值（用于后续计算 Hessian）
activations = defaultdict(int)

# 存储每一层的 Hessian 矩阵（累加值）
hess = defaultdict(float)

# save_activations: 前向传播时的钩子函数
# autograd-lib 在每层前向传播后调用此函数
# A: 该层的输入张量，(batch, input_dim)
# _: 占位符（本函数不需要输出值）
def save_activations(layer, A, _):
    """保存每一层的输入激活值，供后续 Hessian 计算使用"""
    # 对于 MathRegressor:
    #   layer 1 (Linear 1→128): A 形状 (6, 1)
    #   layer 2 (Linear 128→1): A 形状 (6, 128)
    activations[layer] = A

# compute_hess: 反向传播时的钩子函数
# autograd-lib 在反向传播 Hessian 时调用
# B: 该层的反向传播值
def compute_hess(layer, _, B):
    """根据前向激活值和反向传播值计算该层的 Hessian 近似"""
    A = activations[layer]
    # torch.einsum('nl,ni->nli', B, A): 对每个 batch 样本做外积(outer product)
    #   B 形状: (batch, output_dim)
    #   A 形状: (batch, input_dim)
    #   结果形状: (batch, output_dim, input_dim)
    #   这等价于 ∂L/∂w 的每个元素（即损失对权重的梯度）
    BA = torch.einsum('nl,ni->nli', B, A)

    # 完整黑塞矩阵: 再对 BA 做一次外积并累加
    # torch.einsum('nli,nkj->likj', BA, BA):
    #   结果形状: (output_dim, input_dim, output_dim, input_dim)
    #   这就是该层近似的黑塞矩阵（四维张量形式）
    # 每个 batch 的外积累加 = 对 batch 求和
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)


def compute_minimum_ratio(model, criterion, train, target):
    """
    计算极小值比例 = 正特征值数量 / 总特征值数量

    返回 0~1 之间的值:
      接近 1 → 大部分方向向上弯曲 → 接近局部极小值
      接近 0 → 大部分方向向下弯曲 → 接近局部极大值
      0.5 左右 → 鞍点

    判断规则:
      正特征值比例 > 0.5 且 梯度范数 < 1e-3 → 局部极小值类点
      正特征值比例 ≤ 0.5 且 梯度范数 < 1e-3 → 鞍点
    """
    model.zero_grad()

    # 第一步: 前向传播，捕获各层输入激活值
    # autograd_lib.module_hook(save_activations): 上下文管理器
    # 进入后，所有注册过的模块都会自动调用 save_activations 记录输入
    with autograd_lib.module_hook(save_activations):
        output = model(train)
        loss = criterion(output, target)

    # 第二步: 反向传播 Hessian，计算各层的 Hessian 矩阵
    # autograd_lib.module_hook(compute_hess): 激活 Hessian 计算钩子
    # autograd_lib.backward_hessian(output, loss='LeastSquares'):
    #   对均方误差(MSE)损失计算 Hessian
    #   与 loss.backward() 不同，它计算的是二阶导数而非一阶导数
    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(output, loss='LeastSquares')

    # 第三步: 计算每层 Hessian 的特征值，统计正特征值比例
    layer_hess = list(hess.values())
    minimum_ratio = []

    for h in layer_hess:
        # h 是四维张量 (output_dim, input_dim, output_dim, input_dim)
        size = h.shape[0] * h.shape[1]  # 展平后的参数量
        h = h.reshape(size, size)        # 展平为二维方阵 (param_count, param_count)

        # torch.linalg.eigh: 计算实对称矩阵的特征值分解
        # 返回 (eigenvalues, eigenvectors)
        # eigenvalues 按升序排列（从小到大）
        # 注意: 原始 Colab 代码使用 torch.symeig（已弃用），eigh 是其替代
        h_eig = torch.linalg.eigh(h).eigenvalues

        num_greater = torch.sum(h_eig > 0).item()  # 正特征值个数
        ratio = num_greater / len(h_eig)            # 正特征值占比
        minimum_ratio.append(ratio)

    ratio_mean = np.mean(minimum_ratio)  # 各层比例的平均值
    return ratio_mean


# ============================================================
# 6. 主函数：输出梯度范数和极小值比例
# ============================================================
def main(model, train, target):
    """计算并打印梯度范数和极小值比例"""
    # nn.MSELoss: 均方误差损失（回归任务的标准损失）
    # 公式: loss = mean((y_pred - y_true)²)
    # 注意: 任务 2-1 音素分类用的是 CrossEntropyLoss（分类），这里是 MSE（回归）
    criterion = nn.MSELoss()

    gradient_norm = compute_gradient_norm(model, criterion, train, target)
    minimum_ratio = compute_minimum_ratio(model, criterion, train, target)

    print(f'\n梯度范数 (Gradient Norm): {gradient_norm:.6f}')
    print(f'极小值比例 (Minimum Ratio): {minimum_ratio:.6f}')

    return gradient_norm, minimum_ratio


if __name__ == '__main__':
    # 设置随机种子，确保结果可复现
    torch.manual_seed(0)

    # 清空之前可能残留的激活值和 Hessian 记录
    activations.clear()
    hess.clear()

    # 计算
    grad_norm, min_ratio = main(model, train, target)

    # ============================================================
    # 7. 判断结果
    # ============================================================
    # 作业规定:
    #   梯度范数 < 1e-3 视为"梯度为零"
    #   极小值比例 > 0.5 且 梯度为零 → 局部极小值类点
    #   极小值比例 ≤ 0.5 且 梯度为零 → 鞍点
    #   梯度范数 ≥ 1e-3 → 以上皆非
    # ============================================================
    print(f'\n{"="*50}')
    print('判断结果：')

    if grad_norm < 1e-3:
        # 梯度接近零，继续通过 Hessian 判断
        if min_ratio > 0.5:
            print('  → 局部极小值类点 (local minima like)')
            print('    原因: 梯度几乎为零，且大部分特征值为正')
        else:
            print('  → 鞍点 (saddle point)')
            print('    原因: 梯度几乎为零，但正负特征值混合')
    else:
        print('  → 以上皆非 (none of the above)')
        print('    原因: 梯度范数较大，模型仍在下降')

    print(f'{"="*50}')

    # 把这个答案写入文件，方便提交
    if grad_norm < 1e-3:
        answer = 'local minima like' if min_ratio > 0.5 else 'saddle point'
    else:
        answer = 'none of the above'

    with open('hessian_answer.txt', 'w') as f:
        f.write(f'Student ID: {student_id}\n')
        f.write(f'Gradient Norm: {grad_norm:.6f}\n')
        f.write(f'Minimum Ratio: {min_ratio:.6f}\n')
        f.write(f'Answer: {answer}\n')

    print(f'\n答案已保存到 hessian_answer.txt')
