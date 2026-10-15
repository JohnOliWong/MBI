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
    def __init__(self, emb_size, dropout, bias=False):
        self.dropout = dropout
        super().__init__()

        # kernel size for pooling
        # outputs_size = (B, E, 1, 25)
        pooling_kernel = [4, 1, 5]

        # emb_size = 64
        # outputs_size = (B, E, 30, 500 / 4)
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (B, E, 1, 125 / 1)
        # kernel_size = (channel, 1)
        self.eeg_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(30, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (B, E, 1, 125 / 5)
        self.eeg_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
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
        # outputs_size = (64, 6)
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout

        # outputs_size = (72, 25 / 2)
        self.nirs_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (1, 12 / 1)
        # kernel_size = (channel, 1)
        self.nirs_block2 = nn.Sequential(
            DWSConv(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(72, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (1, 12 / 2)
        self.nirs_block3 = nn.Sequential(
            nn.Conv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
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


class TemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.eeg_temporal_projection = EEGTemporalConvLayer(emb_size, dropout)
        self.nirs_temporal_projection = NIRSTemporalConvLayer(emb_size, dropout)

    def forward(self, eeg, nirs):
        temporal_eeg_features = self.eeg_temporal_projection(eeg)
        temporal_nirs_features = self.nirs_temporal_projection(nirs)

        return temporal_eeg_features, temporal_nirs_features


class Positional_Embedding(nn.Module):
    def __init__(self, channels, emb_size, device):
        super().__init__()
        self.channels = channels + 1
        self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), dtype=torch.float32, device=device),
                                    requires_grad=True)

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

    @property
    def self_attention_weights(self):
        return self.attention_weights


class Transformer(nn.Module):
    # calling various transformer functions
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
                 self_dropout, cross_dropout):
        super().__init__()
        self.eeg_nirs_temporal_spatial_attention_weights = None
        self.eeg_temporal_spatial_attention_weights = None
        self.eeg_temporal_attention_weights = None
        self.eeg_spatial_attention_weights = None
        self.nirs_spatial_attention_weights = None
        self.mask = [channels[0] + 1, channels[1] + 1]

        self.eeg_temporal_encoder = Transformer_Encoder(depth, query_size, key_size, value_size, emb_size,
                                                           num_heads, channels[0], expansion, cross_dropout, device)

        self.nirs_temporal_encoder = Transformer_Encoder(depth, query_size, key_size, value_size, emb_size,
                                                            num_heads, channels[1], expansion, cross_dropout, device)

    def forward(self, temporal_eeg, temporal_nirs):
        eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
        nirs_temporal_outputs = self.nirs_temporal_encoder(temporal_nirs)
        return eeg_temporal_outputs[:, 0], nirs_temporal_outputs[:, 0]

    @property
    def get_eeg_attention_weights(self):
        return [self.eeg_temporal_attention_weights, self.eeg_spatial_attention_weights]

    @property
    def get_nirs_spatial_attention_weights(self):
        return [self.nirs_spatial_attention_weights, ]

    @property
    def get_cross_attention_weights(self):
        return [self.eeg_nirs_temporal_spatial_attention_weights, self.eeg_temporal_spatial_attention_weights]


# only temporal convolution is used in this script
class Encoder_EF(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, expansion, conv_dropout,
                 self_dropout, cross_dropout, cls_dropout, num_classes, device):
        super().__init__()
        self.temporal_conv_layer = TemporalConvLayer(emb_size, conv_dropout)
 
        with torch.no_grad():
            eeg, nirs = torch.randn(1, 30, 4000), torch.randn(1, 72, 200)
            eeg_token, nirs_token = self.temporal_conv_layer(eeg, nirs)
            channels = [eeg_token.shape[-1], nirs_token.shape[-1]]

        self.transformer = Transformer(depth, query_size, key_size, value_size, emb_size, num_heads, channels,
                                       expansion, device, self_dropout, cross_dropout)

    def forward(self, eeg, nirs):
        temporal_eeg, temporal_nirs = self.temporal_conv_layer(eeg, nirs)
        temporal_eeg = temporal_eeg.squeeze(-2).permute(0, 2, 1)
        temporal_nirs = temporal_nirs.squeeze(-2).permute(0, 2, 1)
        eeg_token, nirs_token = self.transformer(temporal_eeg, temporal_nirs)

        return eeg_token, nirs_token