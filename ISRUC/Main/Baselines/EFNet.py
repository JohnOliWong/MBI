import torch
import torch.nn as nn
import torch.nn.functional as F


# 26.09.24改进：删除分类头结尾的 Sigmoid
# ISRUC 适配：第二模态由 emg（静态特征）改为 EMG（3 通道时间序列）。
#   EEG 分支 spatial 卷积核按通道数自适应：ISRUC 仅 6 通道，用 (6,8) 一次性塌缩；
#   其余数据集（SEED 62 / EF 30）用原 (8,8)。

class EF_Net(nn.Module):
    def __init__(self, num_classes, mode, eeg_time=None):
        super(EF_Net, self).__init__()
        self.num_classes = num_classes
        self.mode = mode

        # ---- 数据集维度配置 ----
        if mode == 'ISRUC':
            self.eeg_channels = 6      # EEG: F3,C3,O1,F4,C4,O2
            self.eeg_time = 6000 if eeg_time is None else eeg_time
            self.emg_channels = 3      # EMG: X1,X2,X3（下颌）
        elif mode in ('SEEDIV', 'SEEDV'):
            self.eeg_channels = 62
            self.eeg_time = 4000 if eeg_time is None else eeg_time
            self.emg_channels = 31 if mode == 'SEEDIV' else 33
        elif mode in ('EFMI', 'EFMA'):
            self.eeg_channels = 30
            self.eeg_time = 4000 if eeg_time is None else eeg_time
            self.emg_channels = 72
        elif mode == 'EFWG':
            self.eeg_channels = 30
            self.eeg_time = 2000 if eeg_time is None else eeg_time
            self.emg_channels = 72
        else:
            raise ValueError(f'unsupported mode: {mode}')

        # ---- EEG 分支 ----
        self.eeg_conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(1, 16)), nn.ReLU())
        self.eeg_conv2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=(1, 16)), nn.ReLU())
        self.eeg_pool1 = nn.MaxPool2d(kernel_size=(1, 16))
        self.eeg_dropout1 = nn.Dropout(0.5)
        self.eeg_bn1 = nn.BatchNorm2d(8)

        # spatial 卷积核：ISRUC 仅 6 通道，用 (6,8) 塌缩；其余用 (8,8)
        if mode == 'ISRUC':
            self.eeg_conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(self.eeg_channels, 8)), nn.ReLU())
            self.eeg_conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(1, 8)), nn.ReLU())
            self.eeg_pool2 = nn.MaxPool2d(kernel_size=(1, 8))
        else:
            self.eeg_conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(8, 8)), nn.ReLU())
            self.eeg_conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(8, 8)), nn.ReLU())
            self.eeg_pool2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.eeg_dropout2 = nn.Dropout(0.5)
        self.eeg_bn2 = nn.BatchNorm2d(16)

        # 动态计算 EEG 分支 flatten 维度
        x = torch.randn(1, 1, self.eeg_channels, self.eeg_time)
        x = self.eeg_conv1(x)
        x = self.eeg_conv2(x)
        x = self.eeg_conv2(x)
        x = self.eeg_pool1(x)
        x = self.eeg_conv4(x)
        x = self.eeg_conv5(x)
        x = self.eeg_conv5(x)
        x = self.eeg_pool2(x)
        self.eeg_flat_dim = x.contiguous().view(1, -1).size(1)
        self.eeg_fc1 = nn.Sequential(nn.Linear(self.eeg_flat_dim, 256), nn.ReLU())
        self.eeg_dropout3 = nn.Dropout(0.5)
        self.eeg_fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        # ---- EMG 分支（原 emg 分支）----
        self.emg_conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(self.emg_channels, 1), bias=False),
            nn.BatchNorm2d(8), nn.ReLU())
        self.emg_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(16), nn.ReLU())
        if mode == 'ISRUC':
            # EMG 是 6000 点时间序列，保留 16 个时间 bin（约 1.9s/bin）以保留肌张力包络
            self.emg_pool = nn.AdaptiveAvgPool2d((1, 16))
            self.emg_fc1 = nn.Sequential(nn.Linear(16 * 16, 128), nn.ReLU())
        else:
            # 原 emg 分支：静态特征，直接压到 1
            self.emg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.emg_fc1 = nn.Sequential(nn.Linear(16, 128), nn.ReLU())
        self.emg_dropout = nn.Dropout(0.5)

        # ---- fusion ----
        self.combined_fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())  # 128(eeg) + 128(emg)
        self.combined_dropout = nn.Dropout(0.5)
        self.combined_fc2 = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, self.num_classes))

    def forward(self, eeg, emg):
        # EEG pathway
        e = self.eeg_conv1(eeg)
        e = self.eeg_conv2(e)
        e = self.eeg_conv2(e)
        e = self.eeg_pool1(e)
        e = self.eeg_dropout1(e)
        e = self.eeg_bn1(e)

        e = self.eeg_conv4(e)
        e = self.eeg_conv5(e)
        e = self.eeg_conv5(e)
        e = self.eeg_pool2(e)
        e = self.eeg_dropout2(e)
        e = self.eeg_bn2(e)
        e = e.contiguous().view(e.size(0), -1)

        e = self.eeg_fc1(e)
        e = self.eeg_dropout3(e)
        e = self.eeg_fc2(e)

        # EMG pathway
        f = self.emg_conv1(emg)
        f = self.emg_conv2(f)
        f = self.emg_pool(f)
        f = self.emg_dropout(f)
        f = f.contiguous().view(f.size(0), -1)
        f = self.emg_fc1(f)

        # Combined pathway
        combined = torch.cat((e, f), dim=1)
        z = self.combined_fc1(combined)
        z = self.combined_dropout(z)
        z = F.normalize(z, p=2, dim=1)  # L2 normalization
        z = self.combined_fc2(z)
        output = self.output_layer(z)

        return output
