import torch
from torch import nn

from EFBook.Encoder_EF import Encoder_EF
from EFBook.Encoder_Common import Encoder_Common
from EFBook.EF_Book_VQ import Quantization, EGA, Classifier, Cosine_Loss


# SEED 版架构消融 EFBook：与 Main/EFBook/EF_Book_VQ.EF_Book 结构一致（SEEDIV/SEEDV，eeg+eye），
# 仅增加四个开关，分别对应四种消融：
#   use_ot_loss=False           -> OT（去掉 ot_loss）
#   use_disentangling=False     -> DECOUPLE（去掉 disentangling_loss）
#   use_private_codebook=False  -> PC（去掉 private codebook）
#   use_shared_codebook=False   -> SC（去掉 shared codebook）
# 底层组件（Quantization/EGA/Classifier/Encoder_*）从本目录的独立副本 Ablation/EFBook import（与 EF/Ablation 一致）。
#
# 四个开关全开时与 Main 的训练配置一致：loss = 10 * disentangling + 0.05 * codebook_loss。
# 注意：SEED 的 Quantization 默认 ot_weight=5.0（尚未同步 EF 的 0.05 修正），
# 为保证"全开=主模型"，use_ot_loss=True 时传 5.0；False 时置 0 等价于去掉 OT 项。

class EF_Book_Module(nn.Module):
	def __init__(self, dict_len, emb_size, threshold, num_class, mode, device,
	             eeg_time=None, eye_time=None,
	             use_ot_loss=True, use_disentangling=True,
	             use_private_codebook=True, use_shared_codebook=True):
		super().__init__()
		self.emb_size = emb_size
		self.use_disentangling = use_disentangling
		self.use_private_codebook = use_private_codebook
		self.use_shared_codebook = use_shared_codebook

		if mode == 'SEEDIV':
			self.EEG_Width = 4000 if eeg_time is None else eeg_time
			self.Eye_Width = 5 if eye_time is None else eye_time
			self.EEG_Chan, self.Eye_Chan = 62, 31
		elif mode == 'SEEDV':
			self.EEG_Width = 4000 if eeg_time is None else eeg_time
			self.Eye_Width = 5 if eye_time is None else eye_time
			self.EEG_Chan, self.Eye_Chan = 62, 33
		else:
			raise ValueError(f'unsupported mode: {mode}')
		self.num_TConv = 4
		self.num_SConv = 8

		self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size,
								  emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.3,
								  self_dropout=0.3, cross_dropout=0.3, cls_dropout=0.5,
								  num_classes=num_class, mode=mode, device=device, eeg_time=eeg_time, eye_time=eye_time)
		self.eeg_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.EEG_Width, S_Height=self.EEG_Chan,
											  num_TConv=self.num_TConv, num_SConv=self.num_SConv)
		self.eye_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.Eye_Width, S_Height=self.Eye_Chan,
											  num_TConv=self.num_TConv, num_SConv=self.num_SConv)

		ot_weight = 5.0 if use_ot_loss else 0.0
		self.eeg_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.eye_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.fusion_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.cosine_loss = Cosine_Loss()
		self.ega = EGA(emb_size)
		self.classifier = Classifier(emb_size, num_class)

	def forward(self, eeg, eye, last_batch):
		eeg_p_token, eye_p_token = self.ef_conv(eeg, eye)      # [B, E]
		eeg_s_token = self.eeg_common_conv(eeg)                # [B, E]
		eye_s_token = self.eye_common_conv(eye)                # [B, E]
		disentangling_loss = self.cosine_loss(eeg_s_token, eeg_p_token) + self.cosine_loss(eye_s_token, eye_p_token)

		codebook_loss = 0.0
		quan_eeg = torch.zeros_like(eeg_p_token)
		quan_eye = torch.zeros_like(eye_p_token)
		usage_parts = []

		# 私有码本：EEG / 眼动各自独立码本
		if self.use_private_codebook:
			eeg_quan_outputs = self.eeg_quantizer(eeg_p_token)
			quan_eeg = quan_eeg + eeg_quan_outputs['quan_token']
			codebook_loss = codebook_loss + eeg_quan_outputs['codebook_loss']
			if eeg_quan_outputs.get('usage') is not None:
				usage_parts.append(f'EEG {eeg_quan_outputs["usage"]}')

			eye_quan_outputs = self.eye_quantizer(eye_p_token)
			quan_eye = quan_eye + eye_quan_outputs['quan_token']
			codebook_loss = codebook_loss + eye_quan_outputs['codebook_loss']
			if eye_quan_outputs.get('usage') is not None:
				usage_parts.append(f'Eye {eye_quan_outputs["usage"]}')

		# 共享码本：EGA 跨模态注意力后进入 EEG-眼动共享码本
		if self.use_shared_codebook:
			batch_size = eeg_p_token.shape[0]
			eye_s_token = self.ega(eeg_s_token, eye_s_token)
			fusion_features = torch.cat([eeg_s_token, eye_s_token], dim=0)
			fusion_quan_outputs = self.fusion_quantizer(fusion_features)
			quan_fusion = fusion_quan_outputs['quan_token']
			quan_fusion_eeg, quan_fusion_eye = quan_fusion[:batch_size, :], quan_fusion[batch_size:, :]
			quan_eeg = quan_eeg + quan_fusion_eeg
			quan_eye = quan_eye + quan_fusion_eye
			codebook_loss = codebook_loss + 0.5 * fusion_quan_outputs['codebook_loss']
			if fusion_quan_outputs.get('usage') is not None:
				usage_parts.append(f'Shared {fusion_quan_outputs["usage"]}')

		if last_batch and usage_parts:
			print('Codebook Usage: ' + ' '.join(usage_parts))

		eeg_token = eeg_p_token
		eye_token = eye_p_token
		outputs = self.classifier(eeg_token, eye_token, quan_eeg, quan_eye)

		loss = 0.0
		if self.use_disentangling:
			loss = loss + 10 * disentangling_loss
		loss = loss + 0.05 * codebook_loss

		return {
			'outputs': outputs,
			'loss': loss,
		}
