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


# EEG 时间卷积（kernel (1,15)，pooling [4,1,5]，通道塌缩）
class EEGTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=30, bias=False):
        super().__init__()
        self.dropout = 0.3
        pooling_kernel = [4, 1, 5]

        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )
        self.eeg_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )
        self.eeg_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)
        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        return eeg


# 第二模态时间卷积（kernel (1,3)，pooling 可配，通道塌缩）
class EyeTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, channels=72, pooling_kernel=None, bias=False):
        super().__init__()
        self.dropout = 0.3
        if pooling_kernel is None:
            pooling_kernel = [2, 1, 2]

        self.eye_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )
        self.eye_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(channels, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )
        self.eye_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size), nn.ReLU(), nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eye):
        if eye.ndim == 3:
            eye = eye.unsqueeze(1)
        eye = self.eye_block1(eye)
        eye = self.eye_block2(eye)
        eye = self.eye_block3(eye)
        return eye


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
        self.projection = nn.Sequential(nn.Linear(self.proj_dim, emb_size, bias=bias), nn.Dropout(dropout))
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
            nn.Linear(emb_size, expansion * emb_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size), nn.Dropout(dropout))

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
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout, device):
        super(Transformer_Encoder, self).__init__()
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth
        self.add_token = CLS_Token(emb_size)
        self.positional_embedding = Positional_Embedding(channels, emb_size, device)
        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 Self_Encoder(query_size, key_size, value_size, emb_size, num_heads, expansion, dropout))

    def forward(self, x, mask=None):
        x = self.add_token(x)
        x = self.positional_embedding(x)
        for i, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[i] = blk.attention.attention_weights
        return x


class Encoder_EF(nn.Module):
    """EF 编码器（泛化到 N 模态）：每个模态一条时间卷积 + transformer，返回 N 个 CLS token。"""
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, expansion, conv_dropout,
                 self_dropout, cross_dropout, cls_dropout, num_classes, mode, device, modalities=None, eeg_time=None):
        super().__init__()
        self.modalities, modal_chan, modal_time, modal_pool, modal_is_eeg = self._modal_config(mode, modalities, eeg_time)
        self.num_modalities = len(self.modalities)

        self.temporal_convs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for name in self.modalities:
            c, t, p, is_eeg = modal_chan[name], modal_time[name], modal_pool[name], modal_is_eeg[name]
            if is_eeg:
                conv = EEGTemporalConvLayer(emb_size, conv_dropout, channels=c)
            else:
                conv = EyeTemporalConvLayer(emb_size, conv_dropout, channels=c, pooling_kernel=p)
            with torch.no_grad():
                x = torch.randn(2, c, t)
                n_tok = conv(x).shape[-1]
            self.temporal_convs.append(conv)
            self.transformers.append(Transformer_Encoder(depth, query_size, key_size, value_size, emb_size,
                                                         num_heads, n_tok, expansion, cross_dropout, device))

    @staticmethod
    def _modal_config(mode, modalities, eeg_time):
        # 返回 (模态列表, 通道数 dict, 时间 dict, pooling dict, 是否用 EEG 卷积 dict)
        if mode == 'ISRUC':
            mods = list(modalities) if modalities is not None else ['eeg', 'emg']
            chan = {'eeg': 6, 'emg': 3, 'eog': 2}
            time = {'eeg': 6000 if eeg_time is None else eeg_time, 'emg': 6000, 'eog': 6000}
            pool = {'eeg': [4, 1, 5], 'emg': [4, 1, 5], 'eog': [4, 1, 5]}
            is_eeg = {'eeg': True, 'emg': True, 'eog': True}
        elif mode == 'SEEDIV':
            mods = ['eeg', 'emg']
            chan = {'eeg': 62, 'emg': 31}
            time = {'eeg': 4000 if eeg_time is None else eeg_time, 'emg': 5}
            pool = {'eeg': [4, 1, 5], 'emg': [1, 1, 1]}
            is_eeg = {'eeg': True, 'emg': False}
        elif mode == 'SEEDV':
            mods = ['eeg', 'emg']
            chan = {'eeg': 62, 'emg': 33}
            time = {'eeg': 4000 if eeg_time is None else eeg_time, 'emg': 5}
            pool = {'eeg': [4, 1, 5], 'emg': [1, 1, 1]}
            is_eeg = {'eeg': True, 'emg': False}
        elif mode in ('EFMI', 'EFMA'):
            mods = ['eeg', 'emg']
            chan = {'eeg': 30, 'emg': 72}
            time = {'eeg': 4000 if eeg_time is None else eeg_time, 'emg': 200}
            pool = {'eeg': [4, 1, 5], 'emg': [2, 1, 2]}
            is_eeg = {'eeg': True, 'emg': False}
        elif mode == 'EFWG':
            mods = ['eeg', 'emg']
            chan = {'eeg': 30, 'emg': 72}
            time = {'eeg': 2000 if eeg_time is None else eeg_time, 'emg': 100}
            pool = {'eeg': [4, 1, 5], 'emg': [2, 1, 2]}
            is_eeg = {'eeg': True, 'emg': False}
        else:
            raise ValueError(f'unsupported mode: {mode}')
        return mods, chan, time, pool, is_eeg

    def forward(self, inputs):
        # inputs: dict {name: tensor [B,1,chan,time]}
        tokens = []
        for i, name in enumerate(self.modalities):
            t = self.temporal_convs[i](inputs[name])      # [B, emb, 1, T']
            t = t.squeeze(-2).permute(0, 2, 1)            # [B, T', emb]
            token = self.transformers[i](t)[:, 0]         # CLS token [B, emb]
            tokens.append(token)
        return tokens
