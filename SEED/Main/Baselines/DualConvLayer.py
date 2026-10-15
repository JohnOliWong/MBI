'''
This script is used to achieve the local feature extraction within the modality. 
Three convolutional blocks are utilized respectively in the temporal and spatial dimensions of EEG and eye.
'''

import numpy as np
import torch
from torch import nn as nn

class SWConv2d(nn.Module):
    '''
    SW_conv = Depthwise_conv + Pointwise_conv
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), stride=(1, 1), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, stride=stride, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def channel_shuffle(self, x):
        # grouping
        # [b, num_channels, h, w] -> [b, groups, channels_per_group, h, w]
        groups = self.in_channels
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # channel shuffle
        # x.shape=(batch_size, channels_per_group, groups, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # x = self.channel_shuffle(x)

        return x


class EEGSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=30, bias=False):
        self.dropout = dropout
        super().__init__()

        # emb_size= = 64
        # outputs_size = (channels, 500 / 5)
        pooling_kernel = [5, 2, 5]
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (30, 100 / 2)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (30, 50 / 5)
        self.eeg_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # outputs_size = (1, 10)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((channels, 1))

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)

        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        outputs = self.temporal_pooling(eeg)
        return outputs


class EyeSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=72, bias=False):
        super().__init__()

        # emb_size = 64
        # outputs_size = (channels, 25 / 2)
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout
        self.eye_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (channels, 12 / 1)
        self.eye_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (channels, 12 / 2)
        self.eye_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # outputs_size = (1, 6)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((channels, 1))

    def forward(self, eye):
        eye = eye[:, :, :]
        if eye.ndim == 3:
            eye = eye.unsqueeze(1)
        eye = self.eye_block1(eye)
        eye = self.eye_block2(eye)
        eye = self.eye_block3(eye)
        outputs = self.temporal_pooling(eye)

        return outputs


# EEG Temporal Convolution
class EEGTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=30, bias=False):
        self.dropout = dropout
        super().__init__()

        # kernel size for pooling
        # outputs_size = (64, 25)
        pooling_kernel = [4, 1, 5]

        # emb_size = 64
        # outputs_size = (channels, 500 / 4)
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (1, 125 / 1)
        # kernel_size = (channel, 1)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (1, 125 / 5)
        self.eeg_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eeg):
        if eeg.ndim == 3:
            # insert a new dimension in dim=1, i.e. (60, 30, 4000) -> (60, 1, 30, 4000)
            eeg = eeg.unsqueeze(1)

        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        return eeg


class EyeTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=72, pooling_kernel=None, bias=False):
        super().__init__()
        
        # emb_size = 64
        # outputs_size = (64, 6)
        # SEED-IV 的 eye 时间轴只有 K 段，不降采样（pooling_kernel=[1, 1, 1]）
        if pooling_kernel is None:
            pooling_kernel = [2, 1, 2]
        self.dropout = dropout

        # outputs_size = (channels, 25 / 2)
        self.eye_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (1, 12 / 1)
        # kernel_size = (channel, 1)
        self.eye_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (1, 12 / 2)
        self.eye_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eye):
        eye = eye[:, :, :]
        if eye.ndim == 3:
            eye = eye.unsqueeze(1)
        eye = self.eye_block1(eye)
        eye = self.eye_block2(eye)
        eye = self.eye_block3(eye)

        return eye


class SpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, eeg_channels=30, eye_channels=72):
        super().__init__()
        self.eeg_spatial_projection = EEGSpatialConvLayer(emb_size, dropout, channels=eeg_channels, bias=True)
        self.eye_spatial_projection = EyeSpatialConvLayer(emb_size, dropout, channels=eye_channels, bias=True)

    def forward(self, eeg, eye):
        spatial_eeg_features = self.eeg_spatial_projection(eeg)
        spatial_eye_features = self.eye_spatial_projection(eye)

        return spatial_eeg_features, spatial_eye_features


class TemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, eeg_channels=30, eye_channels=72, eye_pooling=None):
        super().__init__()
        self.eeg_temporal_projection = EEGTemporalConvLayer(emb_size, dropout, channels=eeg_channels)
        self.eye_temporal_projection = EyeTemporalConvLayer(emb_size, dropout, channels=eye_channels, pooling_kernel=eye_pooling)

    def forward(self, eeg, eye):
        temporal_eeg_features = self.eeg_temporal_projection(eeg)
        temporal_eye_features = self.eye_temporal_projection(eye)

        return temporal_eeg_features, temporal_eye_features