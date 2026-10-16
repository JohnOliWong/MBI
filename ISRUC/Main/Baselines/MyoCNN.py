import torch
import torch.nn as nn


# 参考：Tsagkas et al., "On the Use of Deeper CNNs in Hand Gesture Recognition
# Based on sEMG Signals", 2019 IISA（MyoUP 数据集，Myo Armband 8 通道 @200Hz）。
# 原版：把多通道 sEMG 当二维「图像」（8 通道 × 15 时间点@75ms）做小卷积 + 池化 + 全连接，
#        输出 12 类手势（Keras 实现，channels_last）。
# ISRUC 适配（PyTorch）：输入 3 通道下颌 EMG × 6000 点（30s@200Hz）→ 5 类睡眠分期。
#   保留原版「二维卷积 + 池化 + 全连接头(512/128/64)」的思想，但：
#   1) 先用卷积塌缩 3 通道 + 时间降采样（6000 -> ~400）；
#   2) 后续时间卷积 + 池化逐步降采样；
#   3) 自适应池化到固定 8 个时间 bin，接全连接分类。

class MyoCNN(nn.Module):
    def __init__(self, in_channels=3, time=6000, num_classes=5):
        super().__init__()
        # Stage 1：塌缩 in_channels 个通道 + 时间降采样（time -> time/15）
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(in_channels, 15), stride=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        # Stage 2：时间卷积 + 池化（/2）
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(), nn.Dropout(0.15),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        # Stage 3：时间卷积(stride 2) + 池化（/2）
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(), nn.Dropout(0.15),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        # Stage 4：时间卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(), nn.Dropout(0.15),
        )
        # 自适应池化到固定 8 个时间 bin
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        # 全连接头（原版 512/128/64）
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, C, T] 或 [B, 1, C, T]
        if x.ndim == 3:
            x = x.unsqueeze(1)       # [B, 1, C, T]
        x = self.conv1(x)            # [B, 32, 1, T/15]
        x = self.conv2(x)            # [B, 32, 1, T/60]
        x = self.conv3(x)            # [B, 64, 1, T/240]
        x = self.conv4(x)            # [B, 128, 1, T/240]
        x = self.pool(x)             # [B, 128, 1, 8]
        x = x.flatten(1)             # [B, 1024]
        x = self.fc(x)
        return x
