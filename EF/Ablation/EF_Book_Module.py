import torch
from torch import nn

from EFBook.Encoder_EF import Encoder_EF
from EFBook.Encoder_Common import Encoder_Common
from EFBook.EF_Book_VQ import Quantization, EGA, Classifier, Cosine_Loss


# 架构消融版 EFBook：与 EFBook.EF_Book_VQ.EF_Book 结构一致，
# 仅增加四个开关，分别对应四种消融：
#   use_ot_loss=False           -> OT（去掉 ot_loss）
#   use_disentangling=False     -> DECOUPLE（去掉 disentangling_loss）
#   use_private_codebook=False  -> PC（去掉 private codebook）
#   use_shared_codebook=False   -> SC（去掉 shared codebook）
# 底层组件（Quantization/EGA/Classifier/Encoder_*）仍从原文件 import，避免重复维护。

class EF_Book_Module(nn.Module):
	def __init__(self, dict_len, emb_size, threshold, num_class, mode, device,
	             use_ot_loss=True, use_disentangling=True,
	             use_private_codebook=True, use_shared_codebook=True):
		super().__init__()
		self.emb_size = emb_size
		self.use_disentangling = use_disentangling
		self.use_private_codebook = use_private_codebook
		self.use_shared_codebook = use_shared_codebook

		if mode in ('EFMI', 'EFMA'):
			self.EEG_Width = 4000
			self.NIRS_Width = 200
		elif mode == 'EFWG':
			self.EEG_Width = 2000
			self.NIRS_Width = 100
		self.num_TConv = 4
		self.num_SConv = 8

		if mode == 'EFMI':
			self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size, 
							      emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.2,
                 				  self_dropout=0.2, cross_dropout=0.2, cls_dropout=0.3, 
								  num_classes=num_class, mode=mode, device=device)
		elif mode == 'EFMA':
			self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size, 
							      emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.2,
                 				  self_dropout=0.2, cross_dropout=0.2, cls_dropout=0.3, 
								  num_classes=num_class, mode=mode, device=device)
		elif mode == 'EFWG':
			self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size, 
							      emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.2,
                 				  self_dropout=0.2, cross_dropout=0.2, cls_dropout=0.3, 
								  num_classes=num_class, mode=mode, device=device)
		
		self.eeg_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.EEG_Width, S_Height=30,
		                                      num_TConv=self.num_TConv, num_SConv=self.num_SConv)
		self.nirs_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.NIRS_Width, S_Height=72,
		                                       num_TConv=self.num_TConv, num_SConv=self.num_SConv)

		ot_weight = 0.05 if use_ot_loss else 0.0
		self.eeg_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.nirs_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.fusion_quantizer = Quantization(dict_len, emb_size, threshold=threshold, ot_weight=ot_weight)
		self.cosine_loss = Cosine_Loss()
		self.ega = EGA(emb_size)
		self.classifier = Classifier(emb_size, num_class)

	def forward(self, eeg, nirs, last_batch):
		eeg_p_token, nirs_p_token = self.ef_conv(eeg, nirs)  # [B, E]
		eeg_s_token = self.eeg_common_conv(eeg)              # [B, E]
		nirs_s_token = self.nirs_common_conv(nirs)           # [B, E]
		disentangling_loss = self.cosine_loss(eeg_s_token, eeg_p_token) + self.cosine_loss(nirs_s_token, nirs_p_token)

		codebook_loss = 0.0
		quan_eeg = torch.zeros_like(eeg_p_token)
		quan_nirs = torch.zeros_like(nirs_p_token)
		usage_parts = []

		# 私有码本：EEG / fNIRS 各自独立码本
		if self.use_private_codebook:
			eeg_quan_outputs = self.eeg_quantizer(eeg_p_token)
			eeg_quan_usage = eeg_quan_outputs.get('usage')
			quan_eeg = quan_eeg + eeg_quan_outputs['quan_token']
			codebook_loss = codebook_loss + eeg_quan_outputs['codebook_loss']

			nirs_quan_outputs = self.nirs_quantizer(nirs_p_token)
			nirs_quan_usage = nirs_quan_outputs.get('usage')
			quan_nirs = quan_nirs + nirs_quan_outputs['quan_token']
			codebook_loss = codebook_loss + nirs_quan_outputs['codebook_loss']

			if eeg_quan_usage is not None:
				usage_parts.append(f'EEG {eeg_quan_usage}')
			if nirs_quan_usage is not None:
				usage_parts.append(f'fNIRS {nirs_quan_usage}')

		# 共享码本：EGA 跨模态注意力后进入 EEG-fNIRS 共享码本
		if self.use_shared_codebook:
			batch_size = eeg_p_token.shape[0]
			nirs_s_token = self.ega(eeg_s_token, nirs_s_token)
			fusion_features = torch.cat([eeg_s_token, nirs_s_token], dim=0)
			fusion_quan_outputs = self.fusion_quantizer(fusion_features)
			fusion_quan_usage = fusion_quan_outputs.get('usage')
			quan_fusion = fusion_quan_outputs['quan_token']
			quan_fusion_eeg, quan_fusion_nirs = quan_fusion[:batch_size, :], quan_fusion[batch_size:, :]
			quan_eeg = quan_eeg + quan_fusion_eeg
			quan_nirs = quan_nirs + quan_fusion_nirs
			codebook_loss = codebook_loss + 0.5 * fusion_quan_outputs['codebook_loss']
			if fusion_quan_usage is not None:
				usage_parts.append(f'Shared {fusion_quan_usage}')

		if last_batch and usage_parts:
			print('Codebook Usage: ' + ' '.join(usage_parts))

		eeg_token = eeg_p_token
		nirs_token = nirs_p_token
		outputs = self.classifier(eeg_token, nirs_token, quan_eeg, quan_nirs)

		loss = 0.0
		if self.use_disentangling:
			loss = loss + 10 * disentangling_loss
		loss = loss + 0.1 * codebook_loss

		return {
			'outputs': outputs,
			'loss': loss,
		}