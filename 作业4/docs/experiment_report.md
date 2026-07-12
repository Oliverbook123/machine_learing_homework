# 作业4 实验对比报告：说话者分类 (Speaker Classification)

## 1. 任务概述

| 项目 | 说明 |
|------|------|
| 任务类型 | 多分类 (600 类说话者) |
| 输入 | mel-spectrogram 特征 `(T, 40)`，T 可变长 |
| 输出 | 600 类说话者 ID |
| 训练集 | 69,438 条带标签样本 |
| 测试集 | 6,000 条无标签样本 |
| 评估指标 | Top-1 准确率 (Accuracy) |
| 硬件环境 | NVIDIA RTX 3070Ti (8GB 显存) |

---

## 2. 三个等级的方案对比

### 2.1 总览

| 维度 | Simple | Medium | Hard |
|------|--------|--------|------|
| 模型 | `Classifier` (Transformer) | `ClassifierV2` (调参 Transformer) | `ConformerClassifier` (Conformer) |
| 编码层 | `TransformerEncoderLayer` × 2 | `TransformerEncoderLayer` × 4 | `ConformerBlock` × 4 |
| d_model | 80 | 80 | 80 |
| nhead | 2 | 4 | 4 |
| dim_ff | 256 | 512 | 512 |
| 池化方式 | Mean Pooling | Self-Attention Pooling | Self-Attention Pooling |
| 位置编码 | 无 | 正弦余弦编码 | 正弦余弦编码 |
| 训练步数 | 70,000 | 70,000 | 100,000 |
| 公开基线 | 0.82523 | 0.90547 | 0.95404 |

### 2.2 Simple — 基础 Transformer

**模型结构** (`src/models.py:76` `Classifier`)：

```
Input (B, T, 40)
  → Linear(40, 80)                        # 输入投影
  → permute(1,0,2) → (T, B, 80)           # 转置为 Transformer 默认格式
  → TransformerEncoderLayer × 2           # 标准 Transformer 编码
  → transpose(0,1) → (B, T, 80)
  → mean(dim=1) → (B, 80)                 # 均值池化
  → Linear(80,80) → ReLU → Linear(80,600) # 分类头
```

**关键参数**: `d_model=80, nhead=2, dim_feedforward=256, dropout=0.1`

**特点**: 完全复现参考代码，`batch_first=False`，使用 `permute` 转置维度。无位置编码（依赖 Transformer 内部的注意力机制）。

### 2.3 Medium — 调参 Transformer + 自注意力池化

**模型结构** (`src/models.py:140` `ClassifierV2`)：

```
Input (B, T, 40)
  → Linear(40, 80)                        # 输入投影
  → PositionalEncoding                    # 正弦余弦位置编码（新增）
  → TransformerEncoderLayer × 4           # 层数 2→4（新增）
  → SelfAttentionPooling → (B, 80)        # 自注意力池化（替代 mean pooling）
  → Linear(80,80) → ReLU → Dropout → Linear(80,600)
```

**相对 Simple 的改进点**:

| 改进 | 原因 |
|------|------|
| nhead 2→4 | 更多注意力头捕获不同子空间的说话者特征 |
| num_layers 2→4 | 更深网络提升表达能力 |
| dim_ff 256→512 | FFN 中间层更宽，增强非线性拟合 |
| 加入 PositionalEncoding | Transformer 本身无位置感知，注入位置信息 |
| Mean Pooling → Self-Attention Pooling | 学习各帧权重，关注说话者关键帧而非简单平均 |
| 分类头加 Dropout | 抑制过拟合 |

### 2.4 Hard — Conformer

**模型结构** (`src/models.py:286` `ConformerClassifier`)：

```
Input (B, T, 40)
  → Linear(40, 80) + PositionalEncoding
  → ConformerBlock × 4
  → SelfAttentionPooling → (B, 80)
  → Linear(80,80) → ReLU → Dropout → Linear(80,600)
```

**ConformerBlock 内部结构** (`src/models.py:248`)：

```
x → FeedForwardModule(½)     # Macaron FFN，半步残差
  → Multi-Head Self-Attention # 全局注意力（带残差 + LayerNorm）
  → ConvModule               # 局部卷积（带残差）
  → FeedForwardModule(½)     # Macaron FFN，半步残差
  → LayerNorm
```

**ConvModule 内部** (`src/models.py:212`)：

```
LayerNorm → Pointwise Conv(1×1) → GLU → Depthwise Conv(kernel=15)
→ BatchNorm → Swish → Pointwise Conv(1×1) → Dropout → 残差
```

**Conformer 相比 Transformer 的核心优势**:

| 特性 | Transformer | Conformer |
|------|-------------|-----------|
| 全局建模 | 自注意力 | 自注意力 |
| 局部建模 | 无 | 深度可分离卷积 (kernel=15) |
| FFN 结构 | 单个 FFN | 两个 Macaron FFN (各贡献 ½) |
| 归一化 | LayerNorm | LayerNorm + BatchNorm (卷积模块内) |

Conformer 结合了 Transformer 的全局序列建模能力和 CNN 的局部特征提取能力，在语音任务中通常优于纯 Transformer。

---

## 3. 数据处理对比

### 3.1 训练数据切分 (Segment)

所有等级使用相同的 `MyDataset` (`src/data.py:26`)：

- 每条音频特征 `(T, 40)`，若 `T > segment_len(128)` 则随机截取 128 帧片段
- 若 `T ≤ 128` 则保留完整序列，由 `collate_batch` 的 `pad_sequence(padding_value=-20)` 填充对齐
- `padding_value=-20`：对应 `log(10^{-20})`，是 mel-spectrogram 的近似最小值，比用 0 填充更合理

### 3.2 训练/验证划分

- 按 90% / 10% 随机划分（`random_split`，`torch.manual_seed(0)` 保证可复现）
- 训练集: 62,494 条，验证集: 6,944 条

### 3.3 推理数据处理

- 推理使用 `InferenceDataset` (`src/data.py:160`)，加载**完整序列**不切分
- `batch_size=1`，`torch.stack` 对齐
- 输出格式: `[feat_path, speaker_string_id]`（通过 `mapping["id2speaker"]` 将数字标签转回字符串）

---

## 4. 训练策略

### 4.1 训练范式

采用 **step-based 训练**（与参考代码一致），而非 epoch-based：

| 参数 | Simple | Medium | Hard |
|------|--------|--------|------|
| total_steps | 70,000 | 70,000 | 100,000 |
| valid_steps | 2,000 | 2,000 | 2,000 |
| warmup_steps | 1,000 | 1,000 | 1,000 |
| save_steps | 10,000 | 10,000 | 10,000 |

step-based 相比 epoch-based 的优势：在大数据集上验证频率更灵活（每2000步验证一次，不必等整个epoch跑完），且学习率 warmup 以步为单位更自然。

### 4.2 学习率调度

使用 **Warmup + Cosine 退火** (`src/train.py:32`)：

- **Warmup 阶段** (前 1,000 步): 学习率线性从 0 增长到 `1e-3`
- **Cosine 退火阶段**: 学习率按余弦曲线从 `1e-3` 衰减到 0

这是 Transformer 架构的标准学习率策略：
- warmup 防止训练早期梯度爆炸（Transformer 初始化后梯度方差大）
- cosine 退火在训练后期精细调优，逐步收敛到更优极小值

### 4.3 优化器与损失

- **优化器**: AdamW (`lr=1e-3, weight_decay=1e-4`)
  - 选择 AdamW 而非 Adam：AdamW 的解耦权重衰减不被自适应学习率扭曲，正则化效果更干净
  - 选择 AdamW 而非 SGD：Transformer 参数梯度尺度差异大，需要自适应学习率
- **损失函数**: CrossEntropyLoss (无标签平滑)
- **batch_size**: 32 (适配 8GB 显存)

---

## 5. 关键设计决策

### 5.1 为什么 d_model 保持 80 而非增大到 256

最初方案中 Medium/Hard 使用 `d_model=256`，但考虑到 3070Ti 仅 8GB 显存：
- `d_model=256, num_layers=4` 的 Transformer 训练时显存占用约 4-5GB
- 推理时长音频 (T>5000) 会导致注意力矩阵 `(B, T, T)` 显存爆炸
- 最终选择 `d_model=80`，搭配更多层和自注意力池化来弥补容量不足

### 5.2 为什么推理时 batch_size=1

测试集音频长度差异极大（最短几十帧，最长超过 6000 帧）。`torch.stack` 要求同 batch 内所有张量形状一致，`batch_size=1` 避免了填充浪费，同时防止超长序列 OOM。

### 5.3 PositionalEncoding max_len 的修复

最初 `max_len=5000`，但测试集中存在超过 5000 帧的音频，推理时位置编码越界报错。修复为 `max_len=16384`，覆盖所有可能的序列长度。

---

## 6. 实验结果

### 6.1 验证集精度

| 等级 | 验证集精度 | 公开基线 | 差距 | 是否达标 |
|------|-----------|---------|------|---------|
| Simple | 0.7457 | 0.82523 | -0.0796 | **未达标** |
| Medium | 0.7967 | 0.90547 | -0.1088 | **未达标** |
| Hard | 0.8217 | 0.95404 | -0.1319 | **未达标** |

### 6.2 训练过程观察

从训练日志中观察到以下现象：

**Hard 等级（Conformer）训练日志关键节点**：
- Step 92000: 验证精度 0.8217（最佳）
- Step 94000~98000: 验证精度稳定在 0.82 左右
- Step 100000: 验证精度 0.8217（最终）
- 训练集精度波动较大（0.78~0.94），但验证集精度在 80k 步后**已饱和**

这说明 Hard 等级的问题**不是训练步数不足**，而是模型容量瓶颈——在 80k 步后已无法从当前架构中获得更多收益。

**三等级精度递增但增幅递减**：
- Simple → Medium: +0.051（调参 + 自注意力池化有效）
- Medium → Hard: +0.025（Conformer 卷积模块贡献有限，被 d_model 过小限制）

### 6.3 未达标原因分析

三个等级均未达标，核心原因是**模型容量不足**，具体分析：

| 原因 | 影响 | 证据 |
|------|------|------|
| **d_model=80 过小** | 600 类说话者需要更高维特征空间区分 | Hard 在 80k 步后饱和，增加步数无收益 |
| **segment_len=128 过短** | 说话者特征需更长的语音上下文 | 参考代码使用更长 segment 可显著提升 |
| **无标签平滑** | 600 类分类任务中标签平滑有助于泛化 | 损失函数仅用普通 CrossEntropy |
| **无数据增强** | 缺少 SpecAugment 等 mel 域增强 | 训练精度波动大，泛化不足 |
| **Simple 无位置编码** | Transformer 无法感知序列顺序 | 仅依赖注意力，损失位置信息 |

### 6.4 改进方向

| 改进项 | 当前 | 建议 | 预期收益 | 显存影响 |
|--------|------|------|---------|---------|
| d_model | 80 | 128 或 256 | 最大 | +30%~100% |
| segment_len | 128 | 256 | 中 | +2x 计算量 |
| label_smoothing | 无 | 0.1 | 小 | 无 |
| SpecAugment 增强 | 无 | 时间/频率掩码 | 中 | 无 |
| Simple 位置编码 | 无 | 加入 PE | 小 | 无 |
| dim_ff | 256/512 | 1024 | 中 | +30% |

从训练速度看（每2000步约47秒，42 step/s），当前显存有明显余量，可安全增大 d_model。

---

## 7. 代码结构

```
作业4/
├── main.py              # 主入口，按 Simple→Medium→Hard 顺序执行
├── requirements.txt
├── src/
│   ├── config.py        # 设备/路径/超参数（step-based）
│   ├── data.py          # MyDataset + collate_batch + InferenceDataset
│   ├── models.py        # Classifier / ClassifierV2 / ConformerClassifier
│   └── train.py        # warmup+cosine / model_fn / valid / train_model / predict
├── data/
│   └── Dataset/         # uttr-*.pt + metadata.json + mapping.json + testdata.json
├── output/              # 模型权重 + 提交 CSV
└── docs/
    └── experiment_report.md  # 本报告
```

---

## 8. 参考文献

1. Vaswani et al. "Attention Is All You Need", NeurIPS 2017
2. Gulati et al. "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020
3. 李宏毅机器学习 2021 Spring — HW4 说话者分类
4. PyTorch 官方文档 — `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`