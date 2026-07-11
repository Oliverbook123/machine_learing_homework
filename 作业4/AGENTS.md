
# 机器学习 HW4 (Machine Learning HW4)

## 目录 (Outline)
- 任务描述 (Task Description)
- 数据集 (Dataset)
- 数据分割 (Data segmentation)
- Kaggle
- 指南 (Guidelines)

---

## 任务介绍 (Task Introduction)
### 自注意力机制 (Self-attention)
- 由 GOOGLE 在论文《Attention is all you need》中提出。它结合了 RNN（考虑整个序列）和 CNN（并行处理）的优势。
- **主要目标**：学习如何使用 Transformer。

### 回顾：HW2 音素分类 (Phoneme classification)
- **任务**：多分类 (Multiclass Classification)
- 从语音中进行逐帧的音素预测。
- **什么是音素？**
  - 语言中语音声音的单位，可以用来区分一个词和另一个词。
  - 例如：bat/pat, bad/bed
  - 机器学习 (Machine Learning) → M AH SH IH N L ER N IH NG
  - *(图片信息：展示了语音信号如何被切分成帧，并对应到不同的音素标签，如 M, AH, SH, IH, N 等)*

### 本次作业：HW4 说话者分类 (Speaker classification)
- **任务**：多分类 (Multiclass Classification)
- 根据给定的语音预测说话者的类别。
- *(图片信息：展示了两个不同说话者（Speaker 1 和 Speaker 2）的语音波形或特征对比，目标是区分出这段语音属于哪个说话者)*

---

## 数据集 (Dataset)
- **训练集**：69,438 个带有标签的处理过的音频特征。
- **测试集**：6,000 个没有标签的处理过的音频特征。
- **标签**：总共 600 个类别，每个类别代表一个说话者。

### 数据预处理 (Data Preprocessing)
*(参考资料：李宏毅教授 [2020 Spring DLHLP] 语音识别)*

### 数据格式 (Data formats)
*(图片信息：展示了数据目录结构的树状图)*
- **数据目录 (Data Directory)** 包含以下文件：
  - `metadata.json`
  - `testdata.json`
  - `mapping.json`
  - `uttr-{random string}.pt` (多个特征文件)

- **metadata 中的信息**：
  - `"n_mels"`: 梅尔频谱图 (mel-spectrogram) 的维度。
  - `"speakers"`: 一个字典。
    - **Key**: 说话者 ID (speaker ids)。
    - **Value**: `"feature_path"` (特征路径) 和 `"mel_len"` (梅尔长度)。

---

## 训练时的数据分割 (Data segmentation during training)
- **不同长度 (Different length)**：音频特征的长度各不相同。
- **训练时分割 (Segment during training)**：在训练过程中对数据进行分割处理。
- *(图片信息：展示了如何将不同长度的音频特征序列切分成固定长度的 Segment，例如 Segment=2 时的切分示意图)*

---

## 示例代码 (Sample Code)
- **Colab 链接**: [link]

### 基线目标 (Baselines):
- **Simple (简单)**：运行示例代码并了解如何使用 Transformer。
- **Medium (中等)**：了解如何调整 Transformer 的参数。
- **Hard (困难)**：构建 **Conformer**，它是 Transformer 的一种变体。

---

## 要求 (Requirements)

### Simple (简单)
- 使用示例代码构建一个自注意力网络 (self-attention network) 来对说话者进行分类。
- **Simple 公开基线 (public baseline)**: 0.82523

### Medium (中等)
- 修改示例代码中 Transformer 模块的参数。
- **Medium 公开基线 (public baseline)**: 0.90547

### Hard (困难)
- 通过构建 **Conformer** 层来提升性能。
- **Hard 公开基线 (public baseline)**: 0.95404

---

## 评分标准 (Grading)
- **评估指标**：@1 准确率 (Accuracy)。
- Simple baseline (public)：+1 pt (示例代码)
- Simple baseline (private)：+1 pt (示例代码)
- Medium baseline (public)：+1 pt
- Medium baseline (private)：+1 pt
- Hard baseline (public)：+1 pt
- Hard baseline (private)：+1 pt
- 上传代码到 NTU COOL：+4 pts
- **总计：10 分**

---

## 代码提交 (Code Submission)
- **NTU COOL (4分)**
  - 将你的代码和报告压缩为 `<student ID>_hw4.zip`
    - *例如：b06901020_hw4.zip*
  - 我们只能看到你最后一次提交的内容。
  - **不要**提交你的模型或数据集。
  - 如果你的代码不合理（无法运行/有作弊嫌疑），你的学期总成绩将 **乘以 0.9**。

### 你的 .zip 文件应仅包含：
- **Code (代码)**: `.py` 或 `.ipynb` 格式
- **Report (报告)**: `.pdf` 格式（仅限获得 10 分的同学）
- *(图片信息：展示了压缩包内文件结构的示例截图)*


---

## 提示 (Hints)

### 1. 自注意力说话者嵌入 (Self-Attentive Speaker Embeddings): [link]
- *(图片信息：展示了说话者分类系统的架构图，提示将原有的池化层替换为 **Self-attention pooling** (自注意力池化))*

### 2. Conformer: [link]
- *(图片信息：展示了 Conformer 的模块结构图，包含卷积模块和自注意力模块的结合)*

### 3. Additive margin softmax: [link]
- *(图片信息：展示了 Additive Margin Softmax 的公式或概念图，用于增加分类的边界裕度)*

---

## 规定 (Regulation)
- **严禁抄袭**，如果你使用了任何其他资源，必须在参考文献中引用。(＊)
- **严禁**手动修改你的预测文件。
- **不要**与任何生物分享代码或预测文件。
- **不要**使用任何方法每天提交结果超过 5 次。
- **不要**搜索或使用额外的数据或预训练模型。
- 如果你违反上述任何规则，你的最终成绩将 **乘以 0.9**。
- 李教授和助教团队保留更改规则和成绩的权利。

*(＊ 参考：科技部研究人员学术伦理指引)*

---
