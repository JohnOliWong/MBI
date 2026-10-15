import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange


class DWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), stride=(1, 1), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, stride=stride, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# EEG Temporal Convolution
class EEGTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=30, bias=False):
        super().__init__()
        self.dropout = 0.3

        # kernel size for pooling
        pooling_kernel = [4, 1, 5]

        # outputs_size = (B, E, C, T / 4)
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (B, E, 1, T / 4)
        # kernel_size = (channel, 1)
        self.eeg_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (B, E, 1, T / 20)
        self.eeg_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)
        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        return eeg

class EyeTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=72, pooling_kernel=None, bias=False):
        super().__init__()
        self.dropout = 0.3

        # kernel size for pooling
        # SEED-IV 的 eye 时间轴只有 K 段，不降采样（pooling_kernel=[1, 1, 1]）
        if pooling_kernel is None:
            pooling_kernel = [2, 1, 2]
        self.dropout = dropout

        # outputs_size = (B, E, C, T / 2)
        self.eye_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (B, E, C, T / 2)
        self.eye_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (B, E, C, T / 4)
        self.eye_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eye):
        if eye.ndim == 3:
            eye = eye.unsqueeze(1)
        eye = self.eye_block1(eye)
        eye = self.eye_block2(eye)
        eye = self.eye_block3(eye)
        return eye

class TemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, eeg_channels=30, eye_channels=72, eye_pooling=None):
        super().__init__()
        self.eeg_temporal_projection = EEGTemporalConvLayer(emb_size, dropout, channels=eeg_channels)
        self.eye_temporal_projection = EyeTemporalConvLayer(emb_size, dropout, channels=eye_channels, pooling_kernel=eye_pooling)

    def forward(self, eeg, eye):
        temporal_eeg_features = self.eeg_temporal_projection(eeg)
        temporal_eye_features = self.eye_temporal_projection(eye)

        return temporal_eeg_features, temporal_eye_features

class Positional_Embedding(nn.Module):
    def __init__(self, channels, emb_size, device):
        super().__init__()
        self.channels = channels + 1
        self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), 
                                                dtype=torch.float32, device=device), requires_grad=True)

    def forward(self, x):
        x = self.pos_emb + x
        return x

class CLS_Token(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)

    def forward(self, x):
        self.cls_token = self.cls_token.to(x.device)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        return x

class Attention(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
        super().__init__()
        self.proj_dim = emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
        self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
        self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.proj_dim, emb_size, bias=bias),
            nn.Dropout(dropout)
        )
        self.attention_weights = None
        self.scaling = emb_size ** (1 / 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        self.attention_weights = F.softmax(energy / self.scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FFN(nn.Module):
    def __init__(self, emb_size, expansion=2, dropout=0.5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feed_forward(x)

        return x

class Self_Encoder(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
        super(Self_Encoder, self).__init__()

        self.attention = Attention(query_size, key_size, value_size, emb_size, num_heads, dropout)
        self.feed_forward = FFN(emb_size, expansion=forward_expansion, dropout=dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # x is the dominant modality
        res = x
        x = self.norm1(x)
        y = self.attention(x, x, x)
        y = y + res

        res = y
        y2 = self.norm2(y)
        y2 = self.feed_forward(y2)
        y2 = y2 + res

        return y2

class Transformer_Encoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(Transformer_Encoder, self).__init__()
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth
        self.add_token = CLS_Token(emb_size)
        self.positional_embedding = Positional_Embedding(channels, emb_size, device)
        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 Self_Encoder(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                  dropout))

    def forward(self, x, mask=None):
        x = self.add_token(x)
        x = self.positional_embedding(x)
        for i, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[i] = blk.attention.attention_weights
        return x

class Transformer(nn.Module):
    # calling various transformer functions
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
                 self_dropout, cross_dropout):
        super().__init__()
        self.eeg_temporal_encoder = Transformer_Encoder(depth, query_size, key_size, value_size, emb_size,
                                                           num_heads, channels[0], expansion, cross_dropout, device)

        self.eye_temporal_encoder = Transformer_Encoder(depth, query_size, key_size, value_size, emb_size,
                                                            num_heads, channels[1], expansion, cross_dropout, device)

    def forward(self, temporal_eeg, temporal_eye):
        eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
        eye_temporal_outputs = self.eye_temporal_encoder(temporal_eye)
        return eeg_temporal_outputs[:, 0], eye_temporal_outputs[:, 0]

class Encoder_EF(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, expansion, conv_dropout,
                 self_dropout, cross_dropout, cls_dropout, num_classes, mode, device, eeg_time=None, eye_time=None):
        super().__init__()
        if mode == 'SEEDIV':
            eeg_chan, eye_chan = 62, 31
            eeg_time = 4000 if eeg_time is None else eeg_time
            eye_time = 5 if eye_time is None else eye_time
            eye_pooling = [1, 1, 1]
        elif mode == 'SEEDV':
            eeg_chan, eye_chan = 62, 33
            eeg_time = 4000 if eeg_time is None else eeg_time
            eye_time = 5 if eye_time is None else eye_time
            eye_pooling = [1, 1, 1]
        elif mode in ('EFMI', 'EFMA'):
            eeg_chan, eye_chan = 30, 72
            eeg_time = 4000 if eeg_time is None else eeg_time
            eye_time = 200 if eye_time is None else eye_time
            eye_pooling = [2, 1, 2]
        elif mode == 'EFWG':
            eeg_chan, eye_chan = 30, 72
            eeg_time = 2000 if eeg_time is None else eeg_time
            eye_time = 100 if eye_time is None else eye_time
            eye_pooling = [2, 1, 2]
        self.temporal_conv_layer = TemporalConvLayer(emb_size, conv_dropout,
                                                     eeg_channels=eeg_chan, eye_channels=eye_chan,
                                                     eye_pooling=eye_pooling)
 
        with torch.no_grad():
            # batch=2 避免 eye 分支在 eye_time=1 时空间塌缩到 (B, C, 1, 1) 导致 BatchNorm 每个通道只有 1 个值
            eeg, eye = torch.randn(2, eeg_chan, eeg_time), torch.randn(2, eye_chan, eye_time)

            # time / 20, time / 4
            eeg_token, eye_token = self.temporal_conv_layer(eeg, eye)
            channels = [eeg_token.shape[-1], eye_token.shape[-1]]

        self.transformer = Transformer(depth, query_size, key_size, value_size, emb_size, num_heads, channels,
                                       expansion, device, self_dropout, cross_dropout)

    def forward(self, eeg, eye):
        # eeg/eye.shape = bz, 1, channel, time
        temporal_eeg, temporal_eye = self.temporal_conv_layer(eeg, eye)
        
        # temporal_eeg.shape = bz, emb_size, 1, time / 20 > bz, time / 20, emb_size
        # temporal_eye.shape = bz, emb_size, 1, time > bz, time, emb_size
        temporal_eeg = temporal_eeg.squeeze(-2).permute(0, 2, 1)
        temporal_eye = temporal_eye.squeeze(-2).permute(0, 2, 1)

        eeg_token, eye_token = self.transformer(temporal_eeg, temporal_eye)

        return eeg_token, eye_token