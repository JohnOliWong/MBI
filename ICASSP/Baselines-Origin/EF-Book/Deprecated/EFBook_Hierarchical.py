import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
import random

from DualConvLayer import TemporalConvLayer

# positional embedding
class PositionalEmbedding(nn.Module):
	def __init__(self, channels, emb_size, device):
		super().__init__()
		self.channels = channels + 1
		self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), dtype=torch.float32, device=device),
									requires_grad=True)

	def forward(self, x):
		x = self.pos_emb + x
		return x


# induce modality labels, eeg = 1, fnirs = 0, this is to help the model to distinguish data type
class ModalityTypeEmbedding(nn.Module):
	def __init__(self, emb_size, token_type_idx=1):
		super().__init__()
		self.token_type_idx = token_type_idx
		self.type_embedding_layer = nn.Embedding(2, emb_size)

	def forward(self, x, mask):
		# x.shape = b, 1 + eeg_tokens + 1 + nirs_tokens, emb_size
		# mask.shape = [1 + eeg_tokens, 1 + nirs_tokens]
		b, _, emb_size = x.shape
		modality_type_emb = torch.ones(b, mask[0] + mask[1], dtype=torch.long, device=x.device) # torch.ones generate a matrix consists of "1"s
		modality_type_emb[:, mask[0]::] = 0 # mask[0]:: = from mask[0] to the end
		type_emb = self.type_embedding_layer(modality_type_emb)
		x = x + type_emb
		return x


# insert learnable cls tokens
class AddClsToken(nn.Module):
	def __init__(self, emb_size):
		super().__init__()
		self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)

	def forward(self, x):
		self.cls_token = self.cls_token.to(x.device)
		x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1) # cls token is repeated batch_size times and concatenated to the start of x
		return x


# inter-series interaction
class MultiHeadAttention(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
		super().__init__()
		self.emb_size = emb_size
		self.proj_dim = self.emb_size
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

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
		queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
		keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
		values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
		energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
		if mask is not None:
			fill_value = torch.finfo(torch.float32).min
			energy.mask_fill(~mask, fill_value)

		scaling = self.emb_size ** (1 / 2)
		self.attention_weights = F.softmax(energy / scaling, dim=-1)
		att = self.att_drop(self.attention_weights)
		out = torch.einsum('bhal, bhlv -> bhav ', att, values)
		out = rearrange(out, "b h n d -> b n (h d)")
		out = self.projection(out)
		return out


# intra-series modeling
# when called, query = key = value
class MultiHeadSelfAttention(nn.Module):
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


# FFN
# introduces non-linearity and feature augmentation
class FeedForwardBlock(nn.Module):
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


# self-attention + residual connection
class SelfEncoderBlock(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
		super(SelfEncoderBlock, self).__init__()

		self.attention = MultiHeadSelfAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
		self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)

	def forward(self, x):
		# x is dominant modality
		res = x
		x = self.norm1(x)
		y = self.attention(x, x, x) # q = k = v
		y = y + res

		res = y
		y2 = self.norm2(y)
		y2 = self.feed_forward(y2)
		y2 = y2 + res

		return y2


# intra-modal attention
class TransformerSelfEncoder(nn.Module):
	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
				 device):
		super(TransformerSelfEncoder, self).__init__()
		self.blks = nn.Sequential()
		self.attention_weights = [None] * depth
		self.add_token = AddClsToken(emb_size)
		self.positional_embedding = PositionalEmbedding(channels, emb_size, device)
		for i in range(depth):
			self.blks.add_module("block" + str(i),
								 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
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


# intra- and inter- modality encoder
class Transformer(nn.Module):
	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
				 self_dropout, cross_dropout):
		super().__init__()
		self.eeg_nirs_temporal_spatial_attention_weights = None
		self.eeg_temporal_spatial_attention_weights = None
		self.eeg_temporal_attention_weights = None
		self.eeg_spatial_attention_weights = None
		self.nirs_spatial_attention_weights = None
		self.mask = [channels[0] + 1, channels[1] + 1]

		self.eeg_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
														   num_heads, channels[0], expansion, cross_dropout, device)

		self.nirs_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
															num_heads, channels[1], expansion, cross_dropout, device)

	def forward(self, temporal_eeg, temporal_nirs):
		eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
		# print(eeg_temporal_outputs.shape)
		nirs_temporal_outputs = self.nirs_temporal_encoder(temporal_nirs)
		# print(nirs_temporal_outputs.shape)

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


class Pooling(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, factor):
		x = x.float()
		device = x.device
		bottom_dim = x.shape[2]
		top_dim = bottom_dim // factor
		pool = nn.AvgPool1d(kernel_size=factor, stride=factor).to(device)
		projection = nn.Sequential(
			nn.Linear(bottom_dim, top_dim),
			nn.ReLU()
		).to(device)

		x = x.permute(0, 2, 1)
		x = pool(x)
		x = x.permute(0, 2, 1)
		x = projection(x)
		x = x.double()
		return x


class ConditionalFusion(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, e):
		x = x.float()
		e = e.float()
		device = x.device
		time_dim = x.shape[1]
		top_dim = e.shape[2]
		bottom_dim = x.shape[2]
		projection = nn.Sequential(
			nn.Linear(top_dim, bottom_dim),
			nn.ReLU(),
		).to(device)
		attention = nn.MultiheadAttention(embed_dim=bottom_dim, num_heads=bottom_dim).to(device)

		e = e.permute(0, 2, 1)
		e = F.interpolate(e, size=time_dim, mode='linear')
		e = e.permute(0, 2, 1)
		e = projection(e)

		# h = x + projection(e)
		h, _ = attention(
			query=e,
			key=x,
			value=x
		)
		h = h.double()
		return h


class Quantization(nn.Module):
	'''
	args:
	cont_features: input continuous concatenated EEG-fNIRS features [B, T, E]
	quan_features: quantized vectors [B, T, E]
	codebook: a dictionary of prototype vectors [D, E]
	N: number of times each codeword is matched [D,]
	m: summation of continuous features matched to the current codeword [D, E]
	status: reflect if a codeword is active or not [D,]
	'''
	def __init__(self, dict_len, emb_size, decay=0.99, ot_weight=1.0):
		super().__init__()
		self.dict_len = dict_len
		self.emb_size = emb_size
		self.decay = decay
		self.ot_weight = ot_weight

		# initiate subject-specific, modality-invariant codebook
		self.codebook = nn.Parameter(torch.randn(dict_len, emb_size))
		self.register_buffer('N', torch.ones(dict_len))
		self.register_buffer('m', self.codebook.clone())
		self.register_buffer('status', torch.zeros(dict_len))

	def forward(self, x, y):
		time_x = x.shape[1]
		time_y = y.shape[1]
		cont_features = torch.cat([x, y], dim=1) # [B, T', E]

		# compute L2-distance between each feature-codeword pair
		distances = torch.cdist(cont_features, self.codebook)
		quan_idx = torch.argmin(distances, dim=-1)  # [B, T']
		quan_token = self.codebook.data[quan_idx]  # [B, T', E]
		quan_token = cont_features + (quan_token - cont_features).detach()
		
		# codebook variables are only updated during training
		if self.training:
			self.codebook_update(cont_features, quan_idx)
		
		quan_eeg = quan_token[:, :time_x, :]
		quan_nirs = quan_token[:, time_x:, :]

		# difference between continuous and discrete features
		quan_loss = F.mse_loss(quan_token.detach(), cont_features)
		# difference between quantized EEG and fNIRS features
		ot_loss = self.ot_loss(cont_features, quan_idx)
		codebook_loss = quan_loss + ot_loss
		return quan_eeg, quan_nirs, codebook_loss
	
	def codebook_update(self, features, indices):
		with torch.no_grad():
			features = features.float()
			# converts discrete codeword indices to sparse matrix
			one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()  # [B, T', D]
			counts = one_hot.sum(dim=[0, 1])  # [D]
			matched = (counts > 0)
			
			if matched.any(): # .any() returns bool variables
				# summation of fine-grained representations matched to the corresponding codeword
				sum_features = torch.einsum('bte,btd->de', features, one_hot) # [D, E]
				# sum_features = sum_features.T
				self.N[matched] = self.decay * self.N[matched] + (1-self.decay) * counts[matched]
				self.m[matched] = self.decay * self.m[matched] + (1-self.decay) * sum_features[matched, :]
				self.codebook.data[matched] = self.m[matched] / self.N[matched].unsqueeze(1)
				self.status[matched] = 0
			
			inactive = ~matched
			self.status[inactive] += 1
			if (self.status >= 100).any():
				self.codeword_replace()

	def codeword_replace(self):
		inactive = (self.status >= 100)
		if inactive.any() and (self.N > 1).any():
			active_idx = torch.where(self.N > 1)[0]
			random_idx = random.choice(active_idx.cpu().numpy())

			replace_codeword = self.codebook.data[random_idx].clone()
			self.codebook.data[inactive] = replace_codeword
			self.N[inactive] = 1
			self.m[inactive] = self.codebook.data[inactive].clone()
			self.status[inactive] = 0

	def ot_loss(self, features, indices):
		one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()
		# compute the probability of each codeword being matched
		codebook_probs = one_hot.mean(dim=[0, 1])
		# the ideal situation is all codewords have the same chance to be used
		target_probs = torch.ones_like(codebook_probs) / len(self.codebook)
		# utilize KL-divergence to make the actual codeword distribution align with the ideal distribution
		ot_loss = F.kl_div(
			input=torch.log(codebook_probs + 1e-6),
			target=target_probs,
			reduction='batchmean'
		)
		return ot_loss


class Classifier(nn.Module):
	def __init__(self, embed_dim=64, num_heads=4, num_classes=2):
		super().__init__()
		self.attention = nn.MultiheadAttention(embed_dim, num_heads)
		self.fusion = ConditionalFusion()
		self.projection = nn.Sequential(
			nn.Linear(16, 32),
			nn.ReLU(),
			nn.Linear(32, 64)
		)
		self.classifier = nn.Linear(2 * embed_dim, num_classes)
		self.pool = nn.AdaptiveAvgPool1d(1)

	def forward(self, quan_eeg_top, quan_nirs_top, quan_eeg_bottom, quan_nirs_bottom):
		time_dim = quan_eeg_bottom.shape[1]
		eeg_fusion = self.fusion(quan_eeg_bottom, quan_eeg_top)
		nirs_fusion = self.fusion(quan_nirs_bottom, quan_nirs_top)
		nirs_fusion = F.interpolate(nirs_fusion.permute(0, 2, 1), size=time_dim, mode='linear').permute(0, 2, 1)

		features = torch.cat([eeg_fusion, nirs_fusion], dim=-1) # [16, 200, 128]
		features = self.pool(features.permute(0, 2, 1)).squeeze(-1) # [16, 128]
		outputs = self.classifier(features)
		return outputs


class EFBook(nn.Module):
	'''
	feature_T = encoder_T(x)
	quan_T = codebook_T(feature_T)
	feature_B = encoder_B(x, quan_T)
	quan_B  =codebook_B(feature_B)
	x' = decoder(quan_T, quan_B)
	'''
	def __init__(self, depth, query_size, key_size, value_size, dict_len, emb_size, decay, num_heads, expansion, conv_dropout,
				 self_dropout, cross_dropout, cls_dropout, num_classes, device):
		super().__init__()
		# embedding size and dropout rate
		self.temporal_conv = TemporalConvLayer(emb_size, conv_dropout)
 
		with torch.no_grad():
			eeg, nirs = torch.randn(1, 30, 4000), torch.randn(1, 72, 200)
			eeg_token, nirs_token = self.temporal_conv(eeg, nirs)
			channels = [eeg_token.shape[-1], nirs_token.shape[-1]]

		self.pooling = Pooling()
		self.quantizer_top = Quantization(dict_len=512, emb_size=16)
		self.quantizer_middle = Quantization(dict_len=512, emb_size=32)
		self.quantizer_bottom = Quantization(dict_len=512, emb_size=64)
		self.fusion = ConditionalFusion()
		self.classifier = Classifier(emb_size, num_classes)

	def forward(self, eeg, nirs):
		temporal_eeg, temporal_nirs = self.temporal_conv(eeg, nirs) # [16, 64, 1, 200] [16, 64, 1, 50]
		temporal_eeg = temporal_eeg.squeeze(-2).permute(0,2,1)
		temporal_nirs = temporal_nirs.squeeze(-2).permute(0,2,1) # [16, 200, 64] [16, 50, 64]

		eeg_top = self.pooling(temporal_eeg, 4) # [16, 50, 16]
		nirs_top = self.pooling(temporal_nirs, 4) # [16, 12, 16]
		quan_eeg_top, quan_nirs_top, quan_loss_top = self.quantizer_top(eeg_top, nirs_top)

		eeg_middle = self.pooling(temporal_eeg, 2) # [16, 100, 32]
		nirs_middle = self.pooling(temporal_nirs, 2) # [16, 25, 32]
		eeg_middle = self.fusion(eeg_middle, quan_eeg_top)
		nirs_middle = self.fusion(nirs_middle, quan_nirs_top)
		quan_eeg_middle, quan_nirs_middle, quan_loss_middle = self.quantizer_middle(eeg_middle, nirs_middle)

		eeg_bottom = self.fusion(temporal_eeg, quan_eeg_middle)
		nirs_bottom = self.fusion(temporal_nirs, quan_nirs_middle)
		quan_eeg_bottom, quan_nirs_bottom, quan_loss_bottom = self.quantizer_bottom(eeg_bottom, nirs_bottom)  # [16, 200, 64] [16, 50, 64]
		outputs = self.classifier(quan_eeg_middle, quan_nirs_middle, quan_eeg_bottom, quan_nirs_bottom)

		return {
			'outputs': outputs,
			'quan_loss': quan_loss_top + quan_loss_middle + quan_loss_bottom
		}