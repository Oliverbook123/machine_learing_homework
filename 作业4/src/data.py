"""
数据加载与预处理
================
参见参考代码：
  - myDataset:          训练用数据集，mel-spectrogram 切成 segment_len 片段
  - collate_batch:      批量整理，用 pad_sequence(padding_value=-20) 对齐
  - get_dataloader:     90%/10% random_split + DataLoader
  - InferenceDataset:   推理用数据集，加载完整序列不切分
"""

import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.config import DATA_DIR, SEGMENT_LEN, BATCH_SIZE, N_WORKERS


# ═══════════════════════════════════════════════════════════════════
# 1. 训练用数据集
# ═══════════════════════════════════════════════════════════════════

class MyDataset(Dataset):
    """
    训练用数据集，与参考代码 myDataset 结构一致。

    - __init__:  从 mapping.json 加载说话者→数字ID映射，
                从 metadata.json 构建样本列表 [(feature_path, speaker_id), ...]
    - __getitem__: 加载 mel-spectrogram，若长于 segment_len 则随机截取片段，
                   否则保留完整序列（由 collate_batch 填充对齐）
    """

    def __init__(self, data_dir=DATA_DIR, segment_len=SEGMENT_LEN):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # json.load: 将 JSON 文件解析为 Python dict
        # Load the mapping from speaker name to their corresponding id.
        mapping_path = os.path.join(data_dir, "mapping.json")
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        # {"id10473": 0, "id10005": 1, ...}
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)["speakers"]

        # 说话者总数
        self.speaker_num = len(metadata.keys())

        # 构建样本列表：[(feature_path, speaker_id_int), ...]
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # torch.load: 从磁盘加载序列化的 Tensor（mel-spectrogram 特征）
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segment: 若特征长于 segment_len，随机选起点截取；否则保留完整
        if len(mel) > self.segment_len:
            # random.randint(a, b): 返回 [a, b] 内随机整数
            start = random.randint(0, len(mel) - self.segment_len)
            # torch.FloatTensor: 将 list/ndarray 转为浮点张量
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        # 将说话者 ID 转为 long 张量（用于后续 CrossEntropyLoss）
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


# ═══════════════════════════════════════════════════════════════════
# 2. 批量整理函数
# ═══════════════════════════════════════════════════════════════════

def collate_batch(batch):
    """
    批量整理：将一个 batch 中长度不一的 mel 用 pad_sequence 对齐。

    参数:
        batch: list of (mel_Tensor, speaker_Tensor)

    返回:
        mel:      (B, max_len, 40)  pad_sequence 填充后的张量
        speaker:  (B,) long 张量

    pad_sequence(fill_value=-20):
        - log(10^-20) ≈ 非常小的值，作为 mel-spectrogram 的 padding 值
        - 比用 0 填充更合理（mel 是 log 尺度，-20 接近实际最小值）
    """
    mel, speaker = zip(*batch)

    # pad_sequence: 将多个长度不同的 Tensor 沿时间维拼接，
    #   短序列用 padding_value 填充到与最长序列等长
    #   batch_first=True: 输出形状 (B, max_len, feature_dim)
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    speaker = torch.FloatTensor(speaker).long()
    # mel: (batch_size, length, 40)
    return mel, speaker


# ═══════════════════════════════════════════════════════════════════
# 3. 创建 DataLoader
# ═══════════════════════════════════════════════════════════════════

def get_dataloader(data_dir=DATA_DIR, batch_size=BATCH_SIZE, n_workers=N_WORKERS):
    """创建训练/验证 DataLoader。

    返回:
        train_loader, valid_loader, speaker_num
    """
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()

    # random_split: 按 90%/10% 随机划分训练/验证
    from src.config import VAL_RATIO
    trainlen = int((1 - VAL_RATIO) * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]

    # torch.manual_seed: 确保每次划分一致
    torch.manual_seed(0)
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=n_workers,
        pin_memory=True, collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset, batch_size=batch_size,
        drop_last=True, num_workers=n_workers,
        pin_memory=True, collate_fn=collate_batch,
    )

    print(f"  总样本: {len(dataset)}, 训练: {trainlen}, 验证: {len(dataset) - trainlen}")
    print(f"  说话者类别: {speaker_num}")

    return train_loader, valid_loader, speaker_num


# ═══════════════════════════════════════════════════════════════════
# 4. 推理用数据集
# ═══════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    """推理用数据集：加载完整 mel-spectrogram 特征序列（不切分）。"""

    def __init__(self, data_dir=DATA_DIR):
        testdata_path = os.path.join(data_dir, "testdata.json")
        with open(testdata_path, "r") as f:
            metadata = json.load(f)
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


def inference_collate_batch(batch):
    """
    推理批量整理：
      batch_size=1 故每 batch 只有 1 条，torch.stack 安全。
    """
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)