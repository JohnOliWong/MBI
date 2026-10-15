"""EyeFormer 核心剥离版（SEED-IV/SEED-V 眼动分类 baseline）。

从 Baselines/EyeFormer（EyeFormer: Predicting Scanpaths in Free-Viewing Tasks
with Transformer-Guided Reinforcement Learning 的原版仓库）中只保留
"眼动序列 Transformer"这一核心结构，去掉：
  - 视觉编码器 ViT（models/vit.py）与跨模态 cross-attention（原版 image -> scanpath 结构）
  - 预训练阶段（NLL/KL 损失、bert-base-uncased 权重初始化）
  - 强化学习阶段（DTW + saliency reward、self-critical 训练）
  - 对 transformers / timm / fastdtw 等外部依赖

与原版核心的对应关系（原文件 -> 本文件）：
  models/xbert.py  BertEmbeddings    -> EyeFormerEmbeddings   （scan_embeddings: Linear(3,d) 换成 Linear(eye_feats,d)）
  models/xbert.py  BertSelfAttention -> EyeFormerSelfAttention（仅保留 self-attention 分支）
  models/xbert.py  BertSelfOutput    -> EyeFormerSelfOutput
  models/xbert.py  BertAttention     -> EyeFormerAttention
  models/xbert.py  BertIntermediate  -> EyeFormerIntermediate（GELU）
  models/xbert.py  BertOutput        -> EyeFormerOutput
  models/xbert.py  BertLayer         -> EyeFormerLayer        （删除 cross-attention）
  models/xbert.py  BertEncoder       -> EyeFormerEncoder

输入与现有 SEED 数据对齐：eye [B, eye_feats, K]
  SEED-IV：eye_feats=31，SEED-V：eye_feats=33，K=每个样本的 4s 段数
模型内部转置为 [B, K, eye_feats] 作为 token 序列。

causal=True ：保留原版"由历史注视预测下一注视"的自回归注意力核心，分类头取最后一个 token。
causal=False：双向注意力 + [CLS] token（BERT 分类风格），分类头取 CLS。
forward(eeg, eye)：eeg 参数仅为兼容现有训练脚本的统一调用签名，内部不使用。
"""
import math

import torch
import torch.nn as nn


class EyeFormerEmbeddings(nn.Module):
	"""对应原版 xbert.BertEmbeddings：scan_embeddings + position_embeddings + LayerNorm + Dropout"""

	def __init__(self, eye_dim, hidden_size, max_len, dropout, ln_eps=1e-12):
		super().__init__()
		self.scan_embeddings = nn.Linear(eye_dim, hidden_size)
		self.position_embeddings = nn.Embedding(max_len, hidden_size)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=ln_eps)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

	def forward(self, x):
		# x: [B, seq_len, eye_dim]
		seq_length = x.size(1)
		inputs_embeds = self.scan_embeddings(x)
		position_ids = self.position_ids[:, :seq_length]
		embeddings = inputs_embeds + self.position_embeddings(position_ids)
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class EyeFormerSelfAttention(nn.Module):
	"""对应原版 xbert.BertSelfAttention 的 self-attention 分支（无 cross-attn / KV cache / 相对位置）"""

	def __init__(self, hidden_size, num_heads, dropout):
		super().__init__()
		if hidden_size % num_heads != 0:
			raise ValueError(f'hidden_size({hidden_size}) 必须能被 num_heads({num_heads}) 整除')
		self.num_attention_heads = num_heads
		self.attention_head_size = hidden_size // num_heads
		self.all_head_size = num_heads * self.attention_head_size
		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(dropout)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask=None):
		query_layer = self.transpose_for_scores(self.query(hidden_states))
		key_layer = self.transpose_for_scores(self.key(hidden_states))
		value_layer = self.transpose_for_scores(self.value(hidden_states))

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if attention_mask is not None:
			# 与 BertModel forward 中预计算的 mask 相加的方式一致
			attention_scores = attention_scores + attention_mask

		attention_probs = nn.Softmax(dim=-1)(attention_scores)
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		return context_layer.view(*new_context_layer_shape)


class EyeFormerSelfOutput(nn.Module):
	"""对应原版 xbert.BertSelfOutput：dense + dropout + 残差后 LayerNorm（post-LN）"""

	def __init__(self, hidden_size, dropout, ln_eps=1e-12):
		super().__init__()
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=ln_eps)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		return self.LayerNorm(hidden_states + input_tensor)


class EyeFormerAttention(nn.Module):
	"""对应原版 xbert.BertAttention"""

	def __init__(self, hidden_size, num_heads, dropout, ln_eps=1e-12):
		super().__init__()
		self.self = EyeFormerSelfAttention(hidden_size, num_heads, dropout)
		self.output = EyeFormerSelfOutput(hidden_size, dropout, ln_eps)

	def forward(self, hidden_states, attention_mask=None):
		self_outputs = self.self(hidden_states, attention_mask)
		return self.output(self_outputs, hidden_states)


class EyeFormerIntermediate(nn.Module):
	"""对应原版 xbert.BertIntermediate（config.hidden_act='gelu'）"""

	def __init__(self, hidden_size, intermediate_size):
		super().__init__()
		self.dense = nn.Linear(hidden_size, intermediate_size)
		self.intermediate_act_fn = nn.GELU()

	def forward(self, hidden_states):
		return self.intermediate_act_fn(self.dense(hidden_states))


class EyeFormerOutput(nn.Module):
	"""对应原版 xbert.BertOutput：dense + dropout + 残差后 LayerNorm"""

	def __init__(self, hidden_size, intermediate_size, dropout, ln_eps=1e-12):
		super().__init__()
		self.dense = nn.Linear(intermediate_size, hidden_size)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=ln_eps)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		return self.LayerNorm(hidden_states + input_tensor)


class EyeFormerLayer(nn.Module):
	"""对应原版 xbert.BertLayer（删除 cross-attention 分支）"""

	def __init__(self, hidden_size, num_heads, intermediate_size, dropout, ln_eps=1e-12):
		super().__init__()
		self.attention = EyeFormerAttention(hidden_size, num_heads, dropout, ln_eps)
		self.intermediate = EyeFormerIntermediate(hidden_size, intermediate_size)
		self.output = EyeFormerOutput(hidden_size, intermediate_size, dropout, ln_eps)

	def forward(self, hidden_states, attention_mask=None):
		attention_output = self.attention(hidden_states, attention_mask)
		intermediate_output = self.intermediate(attention_output)
		return self.output(intermediate_output, attention_output)


class EyeFormerEncoder(nn.Module):
	"""对应原版 xbert.BertEncoder 的 mode='text' 路径（仅自注意力层堆叠）"""

	def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout, ln_eps=1e-12):
		super().__init__()
		self.layer = nn.ModuleList([
			EyeFormerLayer(hidden_size, num_heads, intermediate_size, dropout, ln_eps)
			for _ in range(num_layers)
		])

	def forward(self, hidden_states, attention_mask=None):
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask)
		return hidden_states


class EyeFormer(nn.Module):
	"""EyeFormer 核心（眼动序列 Transformer 分类器，无预训练/无 RL）。

	Args:
		num_classes: 分类数（SEED-IV=4 / SEED-V=5）
		eye_dim:     眼动特征维数（SEED-IV=31 / SEED-V=33）
		seq_len:     序列长度 K（每个样本的 4s 段数）
		hidden_size: token 隐层维度
		num_layers:  Transformer 层数
		num_heads:   注意力头数（需整除 hidden_size）
		mlp_ratio:   FFN 中间层相对 hidden_size 的倍数
		dropout:     各 dropout 概率
		causal:      True=原版自回归核心（最后 token 分类）；False=双向 + CLS
	"""

	def __init__(self, num_classes, eye_dim, seq_len, hidden_size=64, num_layers=3,
				 num_heads=4, mlp_ratio=4.0, dropout=0.1, causal=True, ln_eps=1e-12):
		super().__init__()
		self.causal = causal
		max_len = seq_len if causal else seq_len + 1   # 非因果时额外一个 CLS 位置
		self.embeddings = EyeFormerEmbeddings(eye_dim, hidden_size, max_len, dropout, ln_eps)
		self.encoder = EyeFormerEncoder(num_layers, hidden_size, num_heads,
										int(hidden_size * mlp_ratio), dropout, ln_eps)
		if not causal:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
		self.dropout = nn.Dropout(dropout)
		self.head = nn.Linear(hidden_size, num_classes)

	@staticmethod
	def get_causal_mask(seq_len, device, dtype):
		"""上三角为 -inf 的因果注意力掩码（原版"下一注视点预测"的自回归方式）"""
		return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)

	def forward(self, eeg, eye):
		# eye: [B, eye_feats, K] -> tokens [B, K, eye_feats]
		# eeg 参数仅为兼容现有训练脚本的统一调用签名，本模型不使用
		x = eye.transpose(1, 2).contiguous()
		if self.causal:
			mask = self.get_causal_mask(x.size(1), x.device, x.dtype)   # [K, K]
			hidden = self.encoder(self.embeddings(x), mask)
			pooled = hidden[:, -1]                                      # 最后 token 聚合全部历史
		else:
			# CLS 位于 embedding 空间（与 BERT 一致）：先 embedding 再拼接
			x = self.embeddings(x)
			cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
			x = torch.cat([cls_tokens, x], dim=1)
			hidden = self.encoder(x)                                   # 双向注意力，无掩码
			pooled = hidden[:, 0]                                       # CLS
		return self.head(self.dropout(pooled))