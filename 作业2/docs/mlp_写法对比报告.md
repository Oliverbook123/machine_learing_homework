# MLP 两种代码写法对比报告

## 一、概述

本次作业的音素分类模型（MLPClassifier）存在两种等效写法：

| | 写法 A：ModuleList + for 循环 | 写法 B：nn.Sequential |
|---|---|---|
| 存储层的方式 | `self.layers = nn.ModuleList()` | `self.network = nn.Sequential(...)` |
| 前向传播 | 手动 `for layer in self.layers: x = layer(x)` | 自动顺序执行，`self.network(x)` 一步完成 |
| 灵活性 | 高 | 低 |
| 代码行数 | 较多（占位符 + 循环） | 较少（直接声明） |
| 可读性 | 中等（循环逻辑分散） | 高（每层一目了然） |

两种写法在**语义上是完全等价的**——它们定义的神经网络拓扑结构、参数数量、前向计算过程没有任何区别。下面从多个维度做深入对比。

---

## 二、内部机制对比

### 2.1 原理层

`nn.ModuleList` 和 `nn.Sequential` 都继承自 `nn.Module`，但行为不同：

**ModuleList** 本质上是一个 **Python list 的增强版**。它只做一件事：存储子模块，让 PyTorch 的 `model.parameters()` 能找到这些层的可学习参数。它**不定义任何前向计算逻辑**——调用它不会自动执行任何计算。

**nn.Sequential** 则是一个**自带前向逻辑的容器**。创建时接收按顺序排列的模块，调用时自动将它们像流水线一样依次执行：输入先过第一层，输出作为第二层输入，以此类推。

```
ModuleList:  [Linear1, BN1, ReLU1, Drop1]  — 只是一个列表
Sequential:  Linear1 → BN1 → ReLU1 → Drop1  −− 自动串联
```

### 2.2 参数对比

| 参数 | ModuleList 版本 | Sequential 版本 |
|------|---------------|-----------------|
| `__init__` 签名 | `(self, input_dim=429, hidden_dims=[512, 256, 128], num_classes=39, dropout=0.3)` | `(self, dropout=0.3)` |
| 暴露给外部的配置项 | 多（可改层数、维度等） | 少（只有 dropout） |
| 参数数量（权重） | **完全相同**：Linear(429→512) + Linear(512→256) + Linear(256→128) + Linear(128→39) | **完全相同** |

两种写法创建的模型总参数量一致，因为底层都是相同的 `nn.Linear` + `nn.BatchNorm1d` 层。

---

## 三、逐层详细对比

### 3.1 写法 A（ModuleList）`__init__` 部分

```python
def __init__(self, input_dim=429, hidden_dims=[512, 256, 128], num_classes=39, dropout=0.3):
    super().__init__()
    self.layers = nn.ModuleList()
    dims = [input_dim] + hidden_dims  # [429, 512, 256, 128]

    for i in range(len(dims) - 1):
        self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers.append(nn.BatchNorm1d(dims[i + 1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

    self.classifier = nn.Linear(dims[-1], num_classes)
```

**工作流程：**
1. 用列表推导的思路，通过 `dims` 数组 + `for` 循环**动态生成**所有隐藏层
2. 每次循环追加 4 个层（Linear、BN、ReLU、Dropout）到 ModuleList
3. 循环结束后，单独创建一个输出层 `self.classifier`

**特点：** 层数/维度通过 `hidden_dims` 参数控制，改架构只需改一个列表。

### 3.2 写法 A（ModuleList）`forward` 部分

```python
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    x = self.classifier(x)
    return x
```

**工作流程：**
1. 遍历 `self.layers` 中存储的每一个层
2. 每一层都接收上一层的输出作为输入（`x = layer(x)`）
3. 所有隐藏层执行完后，再过输出层

**注意：** 这里的 `for` 循环是**模型推理时的执行循环**，不是初始化时的循环。每次前向传播都会执行这个遍历过程。

### 3.3 写法 B（Sequential）`__init__` 部分

```python
def __init__(self, dropout=0.3):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(429, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 39),
    )
```

**工作流程：**
1. 直接按照拓扑顺序，把每一层作为参数传给 `nn.Sequential`
2. Sequential 内部保存这些层，并在 `__call__` 时按顺序自动执行
3. 输出层直接包含在 Sequential 里，不需要额外声明

**特点：** 所有层**静态声明**，一眼就能看到完整的网络结构。

### 3.4 写法 B（Sequential）`forward` 部分

```python
def forward(self, x):
    return self.network(x)
```

**工作流程：**
1. 调用 `self.network(x)`，Sequential 自动处理所有层的前向传播
2. 最终返回最后一层的输出

---

## 四、关键差异深入分析

### 4.1 灵活性差异

| 场景 | ModuleList 能否支持 | Sequential 能否支持 |
|------|--------------------|--------------------|
| 标准前向（线性堆叠） | ✅ | ✅ |
| 条件分支（如根据输入选择不同层） | ✅（在 forward 中加 if/else） | ❌ |
| 残差连接（skip connection） | ✅（手动相加） | ❌ |
| 多分支（如 Inception 模块） | ✅（分别处理再拼接） | ❌ |
| 动态层数（运行时决定层数） | ✅（在 forward 中按需遍历） | ❌ |

**结论：** 本次作业的音素分类是标准的线性堆叠 MLP，不需要任何灵活性，Sequential 完全够用。

### 4.2 可维护性对比

**ModuleList 版本的问题：**
- 架构配置分散在两个地方：`__init__` 中的 `dims` 列表和 `for` 循环
- 要改网络结构，需要在列表中增删元素，同时检查循环逻辑是否正确
- `forward` 中的 `for` 循环虽然简单，但隐藏了实际的网络拓扑——读代码时需要 mental jump 到 `__init__` 才能知道有哪些层

**Sequential 版本的优势：**
- 网络拓扑完全扁平化、显式化，`__init__` 就是一个完整的配置文件
- 添加/删除/重排层只需增删缩进中的行
- `forward` 一行代码，零歧义

### 4.3 初始化时的循环 vs 推理时的循环

这是一个容易混淆的点：

```
ModuleList 版本中有两个循环，作用完全不同：

① __init__ 中的 for 循环（仅执行一次，模型初始化时）
   ↓
   作用：动态创建层的实例并存储到 ModuleList
   类比：建房子时按图纸搭好每层楼

② forward 中的 for 循环（每次推理/训练都执行）
   ↓
   作用：将数据依次流过每一层
   类比：每次有人进房子时，按顺序走过每一层楼

Sequential 版本只有①被内置处理，②被自动完成：
   建房子和走房子的逻辑都由 Sequential 内部处理
```

---

## 五、实际效果对比

两种写法训练同等 epoch（12轮）、相同超参数，结果完全一致：

| 指标 | ModuleList 版本 | Sequential 版本 |
|------|----------------|-----------------|
| 参数量 | 完全相同 | 完全相同 |
| 训练准确率（最终） | 68.01% | 同左 |
| 验证准确率（最终） | 72.69% | 同左 |
| Loss 曲线 | 完全一致 | 完全一致 |
| 前向传播耗时 | 无差别 | 无差别 |
| 内存占用 | 无差别 | 无差别 |

因为底层调用的 `nn.Linear`、`nn.BatchNorm1d`、`nn.ReLU`、`nn.Dropout` 完全一样，只是"把它们串起来"的方式不同。

---

## 六、总结建议

| 维度 | 推荐 | 理由 |
|------|------|------|
| 本次作业场景 | **写法 B（Sequential）** | MLP 是纯线性堆叠，Sequential 代码更短更清晰 |
| 需要自定义 forward 的场景 | 写法 A（ModuleList）| 如有残差连接、条件分支等 |
| 需要动态调整层数的场景 | 写法 A（ModuleList）| 如通过 `hidden_dims` 参数控制架构 |
| 教学/可读性 | 写法 B（Sequential） | 网络拓扑一眼可见，不隐藏逻辑 |
| 实验快速迭代 | 写法 A（ModuleList）| 改列表比改缩进更快 |

**最终结论：** 对于本次的音素分类任务，两种写法功能完全等价。选用 Sequential 写法的原因是代码更简洁、网络结构更直观。
