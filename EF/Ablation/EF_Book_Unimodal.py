import torch
from torch import nn

from EFBook.Encoder_Common import Encoder_Common
from EFBook.EF_Book_VQ import Quantization, AttentionFusion, Cosine_Loss
from EFBook.Encoder_EF import EEGTemporalConvLayer, NIRSTemporalConvLayer, Transformer_Encoder


# 单模态消融版：从训练到测试都只用一个模态。
#   modality='eeg'  -> EEG 单模态
#   modality='nirs' -> fNIRS 单模态
# 结构对齐多模态模型的单模态信号通路：
#   Encoder_Common(common token) + Encoder_EF 单分支(private token) + Quantization + 分类头
#   private token 喂给码本量化，分类头融合 [private_token, quan_token]，
#   loss = 10 * disentangling(common, private) + 0.1 * codebook_loss。

class SingleModalEncoder(nn.Module):
	"""Encoder_EF 的单模态分支：只跑一个模态的 temporal conv + transformer，输出 private token（CLS）。"""
	def __init__(self, emb_size, mode, device, modality, depth=4, expansion=2, conv_dropout=0.2, cross_dropout=0.2):
		super().__init__()
		self.modality = modality
		num_heads = 4 if mode in ('EFMI', 'EFMA') else 2

		if modality == 'eeg':
			self.temporal = EEGTemporalConvLayer(emb_size, conv_dropout)
			if mode in ('EFMI', 'EFMA'):
				dummy = torch.randn(1, 30, 4000)
			elif mode == 'EFWG':
				dummy = torch.randn(1, 30, 2000)
		else:
			self.temporal = NIRSTemporalConvLayer(emb_size, conv_dropout)
			if mode in ('EFMI', 'EFMA'):
				dummy = torch.randn(1, 72, 200)
			elif mode == 'EFWG':
				dummy = torch.randn(1, 72, 100)

		with torch.no_grad():
			out = self.temporal(dummy)
			channels = out.shape[-1]

		self.transformer = Transformer_Encoder(depth, emb_size, emb_size, emb_size, emb_size,
		                                       num_heads, channels, expansion, cross_dropout, device)

	def forward(self, x):
		out = self.temporal(x)
		out = out.squeeze(-2).permute(0, 2, 1)
		out = self.transformer(out)
		return out[:, 0]

class EF_Book_Unimodal(nn.Module):
	def __init__(self, dict_len, emb_size, threshold, num_class, mode, device, modality='eeg'):
		super().__init__()
		self.emb_size = emb_size
		self.modality = modality

		if mode in ('EFMI', 'EFMA'):
			if modality == 'eeg':
				self.T_Width, self.S_Height = 4000, 30
			else:
				self.T_Width, self.S_Height = 200, 72
		elif mode == 'EFWG':
			if modality == 'eeg':
				self.T_Width, self.S_Height = 2000, 30
			else:
				self.T_Width, self.S_Height = 100, 72
		self.num_TConv = 4
		self.num_SConv = 8

		self.common_conv = Encoder_Common(num_class, emb_size, T_Width=self.T_Width, S_Height=self.S_Height,
		                                  num_TConv=self.num_TConv, num_SConv=self.num_SConv)
		self.private_encoder = SingleModalEncoder(emb_size, mode, device, modality)
		
		self.quantizer = Quantization(dict_len, emb_size, threshold=threshold)
		self.fusion = AttentionFusion(emb_size)
		self.cosine_loss = Cosine_Loss()
		self.projector = nn.Sequential(
			nn.Linear(emb_size, emb_size // 2),
			nn.ReLU(),
			nn.Linear(emb_size // 2, num_class),
		)

	def forward(self, x, last_batch=False):
		s_token = self.common_conv(x)
		p_token = self.private_encoder(x)
		disentangling_loss = self.cosine_loss(s_token, p_token)

		quan_outputs = self.quantizer(p_token)
		quan_token = quan_outputs['quan_token']
		codebook_loss = quan_outputs['codebook_loss']
		if last_batch and quan_outputs.get('usage') is not None:
			print(f'Codebook Usage: {self.modality} {quan_outputs["usage"]}')

		fusion = self.fusion([p_token, quan_token])
		outputs = self.projector(fusion)
		return {
			'outputs': outputs,
			'loss': 10 * disentangling_loss + 0.1 * codebook_loss,
		}