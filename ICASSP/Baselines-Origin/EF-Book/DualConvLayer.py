'''
This script is used to achieve the local feature extraction within the modality. 

Three convolutional blocks are utilized respectively in the temporal and spatial dimensions of EEG and fNIRS.

'''

import numpy as np
import torch
from torch import nn as nn

# nirs_chan_sel
channel_seq = np.array(range(72)).tolist()

class SWConv2d(nn.Module):
    '''
    depthwise separable convolution
    SW_conv = Depthwise_conv + Pointwise_conv
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), stride=(1, 1), bias=False):
        super().__init__()
        self.in_channels = in_channels
        # channel-wise convolution
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, stride=stride, bias=bias)
        # maps input channels to output channels
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def channel_shuffle(self, x):
        groups = self.in_channels
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # grouping
        # B, C, H, W -> B, G, C', H, W
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # channel shuffle, exchange dim=1 with dim=2
        x = torch.transpose(x, 1, 2).contiguous() # x.shape=(batch_size, channels_per_group, groups, height, width)
        x = x.view(batch_size, -1, height, width) # reshape x to the original shape

        return x

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # x = self.channel_shuffle(x)

        return x


class EEGSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        self.dropout = dropout
        super().__init__()

        # emb_size= = 64
        # output_size = (30, 4000 / 4)
        pooling_kernel = [4, 2, 5]
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # output_size = (30, 1000 / 2)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # output_size = (30, 500 / 5)
        self.eeg_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # output_size = (30, 1)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((30, 1))

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)

        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        outputs = self.temporal_pooling(eeg)
        return outputs


class NIRSSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        super().__init__()

        # emb_size = 64
        # output_size = (36, 200 / 2)
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout
        self.nirs_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # output_size = (36, 100 / 1)
        self.nirs_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # output_size = (36, 100 / 2)
        self.nirs_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # output_size = (36, 1)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((36, 1))

    def forward(self, nirs):
        nirs = nirs[:, :, :]
        if nirs.ndim == 3:
            nirs = nirs.unsqueeze(1)
        nirs = self.nirs_block1(nirs)
        nirs = self.nirs_block2(nirs)
        nirs = self.nirs_block3(nirs)
        outputs = self.temporal_pooling(nirs)

        return outputs


# EEG Temporal Convolution
class EEGTemporalConvLayer(nn.Module):
    '''
    applies 2D convolution along the temporal and spatial dimensions
    output size: [B, E, 1, T']
    B: batch size
    E: embedding size
    T': compressed temporal dimension
    '''
    def __init__(self, emb_size, dropout, bias=False):
        self.dropout = dropout
        super().__init__()

        # kernel size for pooling
        pooling_kernel = [4, 1, 5]

        # emb_size = 64
        # output_size = (16, 64, 30, 4000 / 4)
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias), # same padding by filling 7 pixels on both sides along W
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # output_size = (16, 64, 1, 1000 / 1)
        # kernel_size = (channel, 1)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(30, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # output_size = (16, 64, 1, 1000 / 5)
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


class NIRSTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        super().__init__()
        
        # emb_size = 64
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout

        # output_size = (16, 64, 72, 200 / 2)
        self.nirs_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # output_size = (16, 64, 1, 100 / 1)
        # kernel_size = (channel, 1)
        self.nirs_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(72, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # output_size = (16, 64, 1, 100 / 2)
        self.nirs_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, nirs):
        nirs = nirs[:, :, :]
        if nirs.ndim == 3:
            nirs = nirs.unsqueeze(1)
        nirs = self.nirs_block1(nirs)
        nirs = self.nirs_block2(nirs)
        nirs = self.nirs_block3(nirs)

        return nirs


class SpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.eeg_spatial_projection = EEGSpatialConvLayer(emb_size, dropout, bias=True)
        self.nirs_spatial_projection = NIRSSpatialConvLayer(emb_size, dropout, bias=True)

    def forward(self, eeg, nirs):
        spatial_eeg_features = self.eeg_spatial_projection(eeg)
        spatial_nirs_features = self.nirs_spatial_projection(nirs)

        return spatial_eeg_features, spatial_nirs_features


class TemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.eeg_temporal_projection = EEGTemporalConvLayer(emb_size, dropout)
        self.nirs_temporal_projection = NIRSTemporalConvLayer(emb_size, dropout)

    def forward(self, eeg, nirs):
        temporal_eeg_features = self.eeg_temporal_projection(eeg)
        temporal_nirs_features = self.nirs_temporal_projection(nirs)

        return temporal_eeg_features, temporal_nirs_features