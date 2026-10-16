import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


# 26.09.24改进：
# 原先分类头每次前向传播都会新建投影层，现改为使用emb_size
# self.fc = nn.ModuleDict({'fc1': None, 'fc2': ..., 'fc3': ...})
# def forward(self, x):
#     self.fc['fc1'] = nn.Linear(dimension, 128).to(x.device)  # 每次前向重建

# Convolution module
# local features are captured with convolution instead of positional embedding
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=60, channels=30):
        super().__init__()

        self.shallownet = nn.Sequential(
            # input_size, output_size, kernel_size, stride
            nn.Conv2d(1, emb_size, (1, 64), (1, 1)), # temporal, kernel_size = (1, time)
            nn.Conv2d(emb_size, emb_size, (channels, 1), (1, 1)), # spatial kernel_size = (channel, 1)，SEED-IV 为 62
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)), # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.25),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# Reisdual Connection
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=6,
                 drop_p=0.25,
                 forward_expansion=4,
                 forward_drop_p=0.25):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class Conformer(nn.Sequential):
    def __init__(self, emb_size, depth, n_classes, channels=30, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, channels),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )