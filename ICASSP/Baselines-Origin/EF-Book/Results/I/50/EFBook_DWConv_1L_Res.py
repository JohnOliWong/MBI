import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
import random


class DWSConv(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(DWSConv, self).__init__()
		self.depth_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels)
		self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

	def forward(self, input):
		x = self.depth_conv(input)
		x = self.point_conv(x)
		return x


class fNIRSNet(torch.nn.Module):
	"""
	fNIRSNet model

	Args:
		num_class: Number of classes.
		DHRConv_width: Width of DHRConv = width of fNIRS signals, e.g. time samples.
		DWConv_height: Height of DWConv = height of 2 * fNIRS channels, and '2' means HbO and HbR.
		num_DHRConv: Number of channels for DHRConv.
		num_DWConv: number of channels for DWConv.
	"""
	def __init__(self, num_class, DHRConv_width, DWConv_height, num_DHRConv=4, num_DWConv=8):
		super(fNIRSNet, self).__init__()
		# DHR Module
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=num_DHRConv, kernel_size=(1, DHRConv_width), stride=1, padding=0)
		self.bn1 = torch.nn.BatchNorm2d(num_DHRConv)

		# Global Module
		self.conv2 = DWSConv(in_channels=num_DHRConv, out_channels=num_DWConv, kernel_size=(DWConv_height, 1))
		self.bn2 = torch.nn.BatchNorm2d(num_DWConv)

		self.fc = torch.nn.Linear(num_DWConv, num_class)
		self.act = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.act(self.bn1(self.conv1(x)))
		x = self.act(self.bn2(self.conv2(x)))
		# x = x.view(x.size()[0], -1)
		# x = self.fc(x)
		return x


class Pooling(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, factor):
		x = x.float()
		device = x.device
		bottom_dim = x.shape[1]
		top_dim = bottom_dim // factor
		pool = nn.AvgPool1d(kernel_size=factor, stride=factor).to(device)
		projection = nn.Sequential(
			nn.Linear(bottom_dim, top_dim),
			nn.ReLU()
		).to(device)

		x = projection(x)
		x = x.double()
		return x


class ConditionalFusion(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, e, x):
		x = x.float()
		e = e.float()
		device = x.device
		top_dim = e.shape[1]
		bottom_dim = x.shape[1]
		projection = nn.Sequential(
			nn.Linear(top_dim, bottom_dim),
			nn.ReLU(),
		).to(device)
		attention = nn.MultiheadAttention(embed_dim=bottom_dim, num_heads=bottom_dim).to(device)

		e = projection(e)
		h, _ = attention(
			query=e,
			key=x,
			value=x
		)
		h = h.double()
		return h


class ResidualQuantization(nn.Module):
	'''
	args:
	cont_features: input continuous concatenated EEG-fNIRS features [B, T, E]
	quan_features: quantized vectors [B, T, E]
	codebook: a dictionary of prototype vectors [D, E]
	N: number of times each codeword is matched [D,]
	m: summation of continuous features matched to the current codeword [D, E]
	status: reflect if a codeword is active or not [D,]
	'''
	def __init__(self, dict_len, emb_size, num_quantizers=3, decay=0.99, ot_weight=1.0):
		super().__init__()
		self.dict_len = dict_len
		self.emb_size = emb_size
		self.num_quantizers = num_quantizers
		self.decay = decay
		self.ot_weight = ot_weight

		# initiate subject-specific, modality-invariant codebook
		self.codebooks = nn.ParameterList([
										nn.Parameter(torch.randn(dict_len, emb_size))
										for _ in range(num_quantizers)
										])
		self.register_buffer('N', torch.ones(num_quantizers, dict_len))
		self.register_buffer('m', torch.stack([codebook.clone() for codebook in self.codebooks]))
		self.register_buffer('status', torch.zeros(num_quantizers, dict_len))

	def forward(self, x, y):
		batch_size = x.shape[0]
		cont_features = torch.cat([x, y], dim=0) # [2 * B, E]
		residual = cont_features
		quan_total = 0
		codebook_loss = 0
		all_idx = []

		for i in range(self.num_quantizers):
			# compute L2-distance between each feature-codeword pair
			distances = torch.cdist(cont_features, self.codebooks[i])
			quan_idx = torch.argmin(distances, dim=-1)  # [2 * B]
			quan_token = self.codebooks[i].data[quan_idx]  # [2 * B, E]
			# quan_token = cont_features + (quan_token - cont_features).detach()

			# residual = residual + (quan_token - residual).detach()
			residual = residual - quan_token
			quan_total = quan_total + quan_token
			all_idx.append(quan_idx)
			
			# codebook variables are only updated during training
			if self.training:
				self.codebook_update(i, residual + quan_token.detach(), quan_idx)
				# self.codebook_update(i, cont_features, quan_idx)
			
			# difference between continuous and discrete features
			# codebook_loss += F.mse_loss(quan_token.detach(), residual + quan_token.detach())
			codebook_loss += F.mse_loss(quan_token, residual.detach())
		
		# quan_total = cont_features + (quan_total - cont_features).detach() # quan_total corresponds to quan_token for vq with single codebook
		quan_total = cont_features - residual
		quan_eeg = quan_total[:batch_size, :]
		quan_nirs = quan_total[batch_size:, :]

		# difference between quantized EEG and fNIRS features
		ot_loss = self.ot_loss(cont_features, all_idx[-1])
		codebook_loss += self.ot_weight * ot_loss

		return quan_eeg, quan_nirs, codebook_loss
	
	def codebook_update(self, level, features, indices):
		with torch.no_grad():
			features = features.float()
			# converts discrete codeword indices to sparse matrix
			one_hot = F.one_hot(indices, num_classes=len(self.codebooks[0])).float()  # [2 * B, D]
			counts = one_hot.sum(dim=[0])  # [D]
			matched = (counts > 0)
			
			if matched.any(): # .any() returns bool variables
				# summation of fine-grained representations matched to the corresponding codeword
				sum_features = torch.einsum('nc,nd->dc', features, one_hot) # [D, E]
				self.N[level, matched] = self.decay * self.N[level, matched] + (1-self.decay) * counts[matched]
				self.m[level, matched] = self.decay * self.m[level, matched] + (1-self.decay) * sum_features[matched, :]
				self.codebooks[level].data[matched] = self.m[level, matched] / self.N[level, matched].unsqueeze(1)
				self.status[level, matched] = 0
			
			inactive = ~matched
			self.status[level, inactive] += 1
			if (self.status[level] >= 100).any():
				self.codeword_replace(level)

	def codeword_replace(self, level):
		inactive = (self.status[level] >= 100)
		if inactive.any() and (self.N[level] > 1).any():
			active_idx = torch.where(self.N[level] > 1)[0]
			random_idx = random.choice(active_idx.cpu().numpy())

			replace_codeword = self.codebooks[level].data[random_idx].clone()
			self.codebooks[level].data[inactive] = replace_codeword
			self.N[level, inactive] = 1
			self.m[level, inactive] = self.codebooks[level].data[inactive].clone()
			self.status[level, inactive] = 0

	def ot_loss(self, features, indices):
		one_hot = F.one_hot(indices, num_classes=len(self.codebooks[0])).float()
		# compute the probability of each codeword being matched
		codebook_probs = one_hot.mean(dim=[0])
		# the ideal situation is all codewords have the same chance to be used
		target_probs = torch.ones_like(codebook_probs) / len(self.codebooks[0])
		# utilize KL-divergence to make the actual codeword distribution align with the ideal distribution
		ot_loss = F.kl_div(
			input=torch.log(codebook_probs + 1e-6),
			target=target_probs,
			reduction='batchmean'
		)
		return ot_loss


class Classifier(nn.Module):
	def __init__(self, emb_size=128, num_classes=2):
		super().__init__()
		self.eeg_weight = nn.Parameter(torch.ones(1))
		self.nirs_weight = nn.Parameter(torch.ones(1))
		
		self.eeg_transform = nn.Linear(emb_size * 2, emb_size)
		self.nirs_transform = nn.Linear(emb_size * 2, emb_size)
		
		self.classifier = nn.Sequential(
			nn.Linear(emb_size * 2, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)
	
	def forward(self, eeg_token, nirs_token, quan_eeg, quan_nirs):
		eeg_feats = torch.cat([eeg_token, quan_eeg], dim=-1)
		nirs_feats = torch.cat([nirs_token, quan_nirs], dim=-1)
		eeg_trans = self.eeg_transform(eeg_feats)
		nirs_trans = self.nirs_transform(nirs_feats)
		
		weights = torch.softmax(torch.stack([self.eeg_weight, self.nirs_weight]), dim=0)
		fused = weights[0] * eeg_trans + weights[1] * nirs_trans
		
		combined = torch.cat([eeg_trans, nirs_trans], dim=-1)
		outputs = self.classifier(combined)
		
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

		self.eeg_conv = fNIRSNet(num_classes, DHRConv_width=4000, DWConv_height=30, num_DHRConv=4, num_DWConv=8)
		self.nirs_conv = fNIRSNet(num_classes, DHRConv_width=200, DWConv_height=72, num_DHRConv=4, num_DWConv=8)
		self.interpolation = torch.nn.Linear(8, emb_size)

		self.pooling = Pooling()
		self.quantizer = ResidualQuantization(dict_len, emb_size)
		self.fusion = ConditionalFusion()
		self.classifier = Classifier(emb_size, num_classes)

	def forward(self, eeg, nirs):
		# encoder
		eeg_token = self.eeg_conv(eeg) # [16, 8, 1, 1] = [B, num_DWConv, 1, 1]
		nirs_token = self.nirs_conv(nirs) # [16, 8, 1, 1]

		# if the output token shape is [16, 2, 1, 1], check that if the full-connection layer has been annoted correctly
		eeg_token = eeg_token.squeeze()
		eeg_token = self.interpolation(eeg_token)
		nirs_token = nirs_token.squeeze()
		nirs_token = self.interpolation(nirs_token) # [B, E]

		quan_eeg, quan_nirs, quan_loss = self.quantizer(eeg_token, nirs_token)
		outputs = self.classifier(eeg_token, nirs_token, quan_eeg, quan_nirs)

		# aggregation by given weights
		# quan_loss = 0
		# alpha = 0.5
		# outputs = alpha * eeg_token + (1 - alpha) * nirs_token

		# aggregation by attention weights
		# quan_loss = 0
		# combined = torch.cat([eeg_token.unsqueeze(1), nirs_token.unsqueeze(1)], dim=1)  # [B, 2, num_classes]
		# attn_weights = torch.softmax(combined.mean(dim=-1), dim=1)  # [B, 2]
		# outputs = (attn_weights.unsqueeze(-1) * combined).sum(dim=1)  # [B, num_classes]

		return {
			'outputs': outputs,
			'quan_loss': quan_loss
		}