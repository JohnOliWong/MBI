import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import trunc_normal_


# 26.09.24改进：
# 修改了从timm导入的写法
# 将输出层的回归头改为输出logits（Regression类名保留）

# weights init
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

# scaled dot product
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: Queries, with dimension [B, L_q, D_q]
        :param k: Keys, with dimension [B, L_k, D_k]
        :param v: Values, with dimension [B, L_v, D_v], generally is k
        :param scale: Scaling factor, a float scalar
        :param attn_mask: Masking, with dimension [B, L_q, L_k]
        :return context: Context
        :return attention: Attention
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_v = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_q = nn.Linear(model_dim, model_dim, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(attn_drop)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, num_heads, -1, dim_per_head).transpose(1, 2) \
            .reshape(batch_size, -1, num_heads * dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

# FFN
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = self.w2(self.act(self.w1(x)))
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim, attn_drop=0.0, proj_drop=0.0, feed_drop=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, attn_drop, proj_drop)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, feed_drop)

    def forward(self, input_k, input_v, input_q, attn_mask=None):
        # self attention
        context, attention = self.attention(input_k, input_v, input_q, attn_mask)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class EmbeddingModule(nn.Module):
    """ Embedding module of EEG/EMG features，可对时间轴降采样（pool_factor>1）"""
    def __init__(self, in_chans, out_chans=64, pool_factor=1, patch_embed_bias=True):
        super(EmbeddingModule, self).__init__()
        self.proj = nn.Conv1d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, bias=patch_embed_bias)
        self.BN = torch.nn.BatchNorm1d(out_chans)
        self.pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor) if pool_factor > 1 else nn.Identity()

    def forward(self, x):
        # x dimension: [batch_size, channels, frequency_band]
        out = F.relu(self.BN(self.proj(x)))
        out = self.pool(out)          # [B, out_chans, time/pool_factor]
        out = out.transpose(2, 1)
        return out

class Regression(nn.Module):
    """ Classification head (原 Regression 层，已改为二分类 logits) """
    def __init__(self, hidden_size, num_classes):
        super(Regression, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.predict = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x dimension: [batch_size, N, C]
        out = self.norm(x)
        out = self.avgpool(out.transpose(1, 2))
        out = torch.flatten(out, 1)
        out = self.predict(out)
        return out

class VigilanceNet(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, attn_drop, proj_drop, feed_drop, num_classes, mode, eeg_time=None, emg_time=None):
        super(VigilanceNet, self).__init__()
        if mode == 'ISRUC':
            eeg_chan, emg_chan = 6, 3   # emg 分支在 ISRUC 下处理 EMG（3 通道）
            eeg_time = 6000 if eeg_time is None else eeg_time
            emg_time = 6000 if emg_time is None else emg_time
        elif mode == 'SEEDIV':
            eeg_chan, emg_chan = 62, 31
            eeg_time = 4000 if eeg_time is None else eeg_time
            emg_time = 5 if emg_time is None else emg_time
        elif mode == 'SEEDV':
            eeg_chan, emg_chan = 62, 33
            eeg_time = 4000 if eeg_time is None else eeg_time
            emg_time = 5 if emg_time is None else emg_time
        elif mode in ('EFMI', 'EFMA'):
            eeg_chan, emg_chan = 30, 72
            eeg_time = 4000 if eeg_time is None else eeg_time
            emg_time = 200 if emg_time is None else emg_time
        elif mode == 'EFWG':
            eeg_chan, emg_chan = 30, 72
            eeg_time = 2000 if eeg_time is None else eeg_time
            emg_time = 100 if emg_time is None else emg_time
        
        # ISRUC 时间轴长（6000），每时间点一个 token 会致 6000² 注意力；降采样到 ~600 token
        pool_factor = 10 if mode == 'ISRUC' else 1

        self.eeg_embed = EmbeddingModule(in_chans=eeg_chan, out_chans=hidden_size, pool_factor=pool_factor)
        self.eeg_pos_embed = nn.Parameter(torch.zeros(1, eeg_time // pool_factor, hidden_size))
        trunc_normal_(self.eeg_pos_embed, std=0.02) # truncated normal distribution to initialze the position embedding

        self.emg_embed = EmbeddingModule(in_chans=emg_chan, out_chans=hidden_size, pool_factor=pool_factor)
        self.emg_pos_embed = nn.Parameter(torch.zeros(1, emg_time // pool_factor, hidden_size))
        trunc_normal_(self.emg_pos_embed, std=0.02)

        self.encoder = EncoderLayer(hidden_size, num_heads, ffn_dim, attn_drop, proj_drop, feed_drop)
        self.predict = Regression(hidden_size, num_classes)
        self.apply(_init_weights)

    def forward(self, eeg, emg):
        eeg = self.eeg_embed(eeg)
        eeg = eeg + self.eeg_pos_embed # avoid in place operation
        eeg, _ = self.encoder(eeg, eeg, eeg) # encoder(k, v, q)

        emg = self.emg_embed(emg)
        emg = emg + self.emg_pos_embed
        emg, _ = self.encoder(emg, emg, emg)

        out, _ = self.encoder(emg, emg, eeg)
        logits = self.predict(out)

        return logits