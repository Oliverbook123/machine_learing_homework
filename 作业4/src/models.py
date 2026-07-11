"""
模型定义
========
包含三个等级的模型：
  - Classifier:               参考代码基础 Transformer（Simple）
  - ClassifierV2:             调参版 + 自注意力池化（Medium）
  - ConformerClassifier:      Conformer 模型（Hard）

输入:  (B, T, 40)  mel-spectrogram 特征
输出:  (B, 600)    说话者分类 logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import N_MELS


# ═══════════════════════════════════════════════════════════════════
# 通用组件
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（详见 hw3/hw4 示例）。

    Transformer 本身没有位置感知能力，需要注入位置信息。
    使用固定（不可学习）的正余弦编码。
    """

    def __init__(self, d_model, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, T, d_model) → x + 位置编码"""
        return x + self.pe[:, :x.size(1)]


class SelfAttentionPooling(nn.Module):
    """
    自注意力池化：学习每个时间步的重要权重，加权求和。

    相比 mean pooling 更关注说话者相关的关键帧。
    """

    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: (B, T, d_model) → pooled: (B, d_model)
        """
        scores = torch.tanh(self.w1(x))
        scores = self.w2(scores)
        alpha = F.softmax(scores, dim=1)
        pooled = torch.sum(alpha * x, dim=1)
        return pooled


# ═══════════════════════════════════════════════════════════════════
# Simple 等级：基础 Transformer（与参考代码 Classifier 一致）
# ═══════════════════════════════════════════════════════════════════

class Classifier(nn.Module):
    """
    参考代码的 Classifier，使用 TransformerEncoderLayer + mean pooling。

    结构:
        Input (B, T, 40)
          → Linear(40, d_model)
          → TransformerEncoderLayer（注意: 非批量优先，需转置）
          → mean pooling over time
          → Linear → ReLU → Linear → (B, n_spks)
    """

    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()

        # 输入投影：40 维 mel → d_model 维
        self.prenet = nn.Linear(N_MELS, d_model)

        # nn.TransformerEncoderLayer: 标准 Transformer 编码器层
        #   d_model: 特征维度
        #   dim_feedforward: FFN 的隐藏维度
        #   nhead: 多头注意力的头数
        #   注意: batch_first=False（默认），所以输入需转置为 (T, B, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout,
        )
        # 堆叠两层 encoder
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # 分类头
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        参数:
            mels: (B, T, 40)
        返回:
            out: (B, n_spks)
        """
        # out: (B, T, d_model)
        out = self.prenet(mels)

        # permute(1, 0, 2): 转为 (T, B, d_model) — TransformerEncoder 默认接受此格式
        out = out.permute(1, 0, 2)
        out = self.encoder(out)

        # 转回 (B, T, d_model)
        out = out.transpose(0, 1)

        # mean pooling: 沿时间维取平均 → (B, d_model)
        stats = out.mean(dim=1)

        out = self.pred_layer(stats)
        return out


# ═══════════════════════════════════════════════════════════════════
# Medium 等级：调参 Transformer + 自注意力池化
# ═══════════════════════════════════════════════════════════════════

class ClassifierV2(nn.Module):
    """
    Medium 等级改进点:
      1. d_model 80→80, nhead 2→4, FFN 256→512
      2. num_layers 2→4
      3. mean pooling → self-attention pooling
      4. 增加 Dropout 抑制过拟合
    """

    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(N_MELS, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # 增大 nhead 和层数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=512, nhead=4, dropout=dropout,
            batch_first=True,  # 用 batch_first 更直观
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 自注意力池化替代 mean pooling
        self.pooling = SelfAttentionPooling(d_model, hidden=128)

        self.dropout = nn.Dropout(dropout)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            self.dropout,
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        参数: mels (B, T, 40)
        返回: (B, n_spks)
        """
        out = self.prenet(mels)
        out = self.pos_enc(out)
        out = self.encoder(out)
        out = self.pooling(out)
        out = self.pred_layer(out)
        return out


# ═══════════════════════════════════════════════════════════════════
# Hard 等级：Conformer
# ═══════════════════════════════════════════════════════════════════

class FeedForwardModule(nn.Module):
    """
    Conformer 的 Macaron-style FFN:
        LayerNorm → Linear → Swish → Dropout → Linear → Dropout

    输出乘 0.5 用于半步残差。
    """

    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.ff(self.norm(x))


class ConvModule(nn.Module):
    """
    Conformer 卷积模块:
        LayerNorm → Pointwise(1×1) Conv → GLU → Depthwise Conv(kernel=15)
        → BatchNorm → Swish → Pointwise Conv → Dropout → 残差
    """

    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # (B, T, d) → (B, d, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = F.glu(x, dim=1)
        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        # 转回 (B, T, d)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    """
    标准 Conformer 块:
        x → FFN(half) → MHSA → Conv → FFN(half) → LayerNorm

    参考: Gulati et al. 2020
    """

    def __init__(self, d_model, nhead, dim_ff, conv_kernel=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dim_ff, dropout)
        self.ffn2 = FeedForwardModule(d_model, dim_ff, dropout)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # FFN(half)
        x = self.ffn1(x)
        # Multi-Head Self-Attention with residual
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + self.attn_dropout(attn_out)
        # Conv (with residual)
        x = self.conv(x)
        # FFN(half)
        x = self.ffn2(x)
        # Final LayerNorm
        x = self.norm(x)
        return x


class ConformerClassifier(nn.Module):
    """
    Hard 等级 Conformer 编码器 + 自注意力池化。

    结构:
        Input (B, T, 40)
          → Linear(40, d_model) + PositionalEncoding
          → ConformerBlock × num_blocks
          → Self-Attention Pooling
          → Linear → ReLU → Linear → (B, n_spks)
    """

    def __init__(self, d_model=80, n_spks=600, num_blocks=4,
                 nhead=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(N_MELS, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_ff, conv_kernel=15, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pooling = SelfAttentionPooling(d_model, hidden=128)
        self.dropout = nn.Dropout(dropout)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            self.dropout,
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        参数: mels (B, T, 40)
        返回: (B, n_spks)
        """
        out = self.prenet(mels)
        out = self.pos_enc(out)
        for block in self.blocks:
            out = block(out)
        pooled = self.pooling(out)
        out = self.pred_layer(pooled)
        return out