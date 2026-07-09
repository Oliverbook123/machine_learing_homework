"""
模型定义
=========
包含两个等级的模型：
  - BasicCNN:     简单 CNN（Easy）
  - ImprovedCNN:  带残差块的改进 CNN（Medium / Hard）
"""

import torch
import torch.nn as nn

from src.config import NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════
# Easy 等级：基础 CNN
# ═══════════════════════════════════════════════════════════════════

class BasicCNN(nn.Module):
    """
    简单的卷积神经网络：
    Conv2d → BN → ReLU → MaxPool → Conv2d → BN → ReLU → MaxPool →
    Conv2d → BN → ReLU → MaxPool → Conv2d → BN → ReLU → MaxPool →
    AdaptiveAvgPool → Dropout → Linear

    输入:  (B, 3, 128, 128)
    输出:  (B, 11)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # nn.Sequential: 按顺序执行，前一层输出自动作为后一层输入
        # nn.Conv2d(in_ch, out_ch, k, padding): 2D 卷积
        # nn.BatchNorm2d: 批归一化，稳定训练
        # nn.ReLU: f(x) = max(0, x)，引入非线性
        # nn.MaxPool2d(k, s): 取 k×k 区域最大值，步长 s，空间尺寸减半
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128→64

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64→32

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8
        )

        # nn.AdaptiveAvgPool2d((1,1)): 全局平均池化，每个通道→一个标量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # nn.Dropout(0.5): 训练时随机丢弃 50% 的神经元，防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 256, 1, 1) → (B, 256)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Medium 等级：带残差块的改进 CNN
# ═══════════════════════════════════════════════════════════════════

class SimpleResBlock(nn.Module):
    """
    简化残差块（Residual Block）。
    output = F(input) + input
    残差连接让梯度可以直接回传，缓解深层网络梯度消失问题。

    参考: He et al. "Deep Residual Learning for Image Recognition", CVPR 2016
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 当通道数或空间尺寸变化时，用 1×1 卷积调整 shortcut 分支
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                           kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual          # 残差连接
        out = self.relu(out)
        return out


class ImprovedCNN(nn.Module):
    """
    带残差块的改进 CNN，4 个阶段逐步增加通道数、降低空间尺寸。
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(     # 64 × 128 × 128
            SimpleResBlock(64, 64),
            SimpleResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(     # 128 × 64 × 64
            SimpleResBlock(64, 128, stride=2),
            SimpleResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(     # 256 × 32 × 32
            SimpleResBlock(128, 256, stride=2),
            SimpleResBlock(256, 256),
        )
        self.layer4 = nn.Sequential(     # 512 × 16 × 16
            SimpleResBlock(256, 512, stride=2),
            SimpleResBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming 正态初始化，适用于 ReLU 激活"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
