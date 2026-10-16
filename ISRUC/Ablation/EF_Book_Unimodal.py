import torch
from torch import nn

from EFBook.Encoder_Common import Encoder_Common
from EFBook.EF_Book_VQ import Quantization, AttentionFusion, Cosine_Loss
from EFBook.Encoder_EF import EEGTemporalConvLayer, EyeTemporalConvLayer, Transformer_Encoder


# SEED 版单模态消融：从训练到测试都只用一个模态。
#   modality='eeg' -> EEG 单模态
#   modality='eye' -> 眼动单模态
# 结构对齐多模态模型的单模态信号通路：
#   Encoder_Common(common token) + Encoder_EF 单分支(private token) + Quantization + 分类头
#   private token 喂给码本量化，分类头融合 [private_token, quan_token]，
#   loss = 10 * disentangling(common, private) + 0.05 * codebook_loss（与 Main 权重一致）。

class SingleModalEncoder(nn.Module):
	"""Encoder_EF 的单模态分支：只跑一个模态的 temporal conv + transformer，输出 private token（CLS）。"""
	def __init__(self, emb_size, mode, device, modality, eeg_time=None, eye_time=None,
	             depth=4, expansion=2, conv_dropout=0.2, cross_dropout=0.2):
		super().__init__()
		self.modality = modality
		num_heads = 4

		if mode == 'SEEDIV':
			eeg_chan, eye_chan, eye_pooling = 62, 31, [1, 1, 1]
			eeg_time = 4000 if eeg_time is None else eeg_time
			eye_time = 5 if eye_time is None else eye_time
		elif mode == 'SEEDV':
			eeg_chan, eye_chan, eye_pooling = 62, 33, [1, 1, 1]
			eeg_time = 4000 if eeg_time is None else eeg_time
			eye_time = 5 if eye_time is None else eye_time
		else:
			raise ValueError(f'unsupported mode: {mode}')

		if modality == 'eeg':
			self.temporal = EEGTemporalConvLayer(emb_size, conv_dropout, channels=eeg_chan)
			# batch=2：避免 eye_time=1 等短时间序列在空间塌缩后 BatchNorm 每个通道只有 1 个值
			dummy = torch.randn(2, eeg_chan, eeg_time)
		else:
			self.temporal = EyeTemporalConvLayer(emb_size, conv_dropout, channels=eye_chan, pooling_kernel=eye_pooling)
			dummy = torch.randn(2, eye_chan, eye_time)

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
	def __init__(self, dict_len, emb_size, threshold, num_class, mode, device, modality='eeg',
	             eeg_time=None, eye_time=None):
		super().__init__()
		self.emb_size = emb_size
		self.modality = modality

		if mode == 'SEEDIV':
			eeg_chan, eye_chan = 62, 31
			eeg_time = 4000 if eeg_time is None else eeg_time
			eye_time = 5 if eye_time is None else eye_time
		elif mode == 'SEEDV':
			eeg_chan, eye_chan = 62, 33
			eeg_time = 4000 if eeg_time is None else eeg_time
			eye_time = 5 if eye_time is None else eye_time
		else:
			raise ValueError(f'unsupported mode: {mode}')

		self.T_Width, self.S_Height = (eeg_time, eeg_chan) if modality == 'eeg' else (eye_time, eye_chan)
		self.num_TConv = 4
		self.num_SConv = 8

		self.common_conv = Encoder_Common(num_class, emb_size, T_Width=self.T_Width, S_Height=self.S_Height,
										  num_TConv=self.num_TConv, num_SConv=self.num_SConv)
		self.private_encoder = SingleModalEncoder(emb_size, mode, device, modality,
												  eeg_time=eeg_time, eye_time=eye_time)

		self.quantizer = Quantization(dict_len, emb_size, threshold=threshold)
		self.fusion = AttentionFusion(emb_size)
		self.cosine_loss = Cosine_Loss()
		# 与 Main 分类头保持一致：Linear-ReLU-Dropout(0.2)-Linear
		self.projector = nn.Sequential(
			nn.Linear(emb_size, emb_size // 2),
			nn.ReLU(),
			nn.Dropout(0.2),
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
			print(f"Codebook Usage: {self.modality} {quan_outputs['usage']}")

		fusion = self.fusion([p_token, quan_token])
		outputs = self.projector(fusion)
		return {
			'outputs': outputs,
			'loss': 10 * disentangling_loss + 0.05 * codebook_loss,
		}
