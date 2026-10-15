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
		# time_x = x.shape[1]
		# time_y = y.shape[1]
		batch_size = x.shape[0]
		cont_features = torch.cat([x, y], dim=0) # [2 * B, E]

		# compute L2-distance between each feature-codeword pair
		distances = torch.cdist(cont_features, self.codebook)
		quan_idx = torch.argmin(distances, dim=-1)  # [2 * B]
		quan_token = self.codebook.data[quan_idx]  # [2 * B, E]
		quan_token = cont_features + (quan_token - cont_features).detach()
		
		# codebook variables are only updated during training
		if self.training:
			self.codebook_update(cont_features, quan_idx)
		
		quan_eeg = quan_token[:batch_size, :]
		quan_nirs = quan_token[batch_size:, :]

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
			one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()  # [2 * B, D]
			counts = one_hot.sum(dim=[0])  # [D]
			matched = (counts > 0)
			
			if matched.any(): # .any() returns bool variables
				# summation of fine-grained representations matched to the corresponding codeword
				sum_features = torch.einsum('nc,nd->dc', features, one_hot) # [D, E]
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
		codebook_probs = one_hot.mean(dim=[0])
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
	def __init__(self, embed_dim=128, num_heads=4, num_classes=2):
		super().__init__()
		self.fusion = ConditionalFusion()
		self.classifier = nn.Linear(2 * embed_dim, num_classes)

	def forward(self, quan_eeg_middle, quan_nirs_middle, quan_eeg_bottom, quan_nirs_bottom):
		eeg_fusion = self.fusion(quan_eeg_middle, quan_eeg_bottom) # [16, 128]
		nirs_fusion = self.fusion(quan_nirs_middle, quan_nirs_bottom) # [16, 128]

		features = torch.cat([eeg_fusion, nirs_fusion], dim=-1) # [16, 256]
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

		self.eeg_conv = fNIRSNet(num_classes, DHRConv_width=4000, DWConv_height=30, num_DHRConv=4, num_DWConv=8)
		self.nirs_conv = fNIRSNet(num_classes, DHRConv_width=200, DWConv_height=72, num_DHRConv=4, num_DWConv=8)
		self.interpolation = torch.nn.Linear(8, emb_size)

		self.pooling = Pooling()
		self.emb_size = emb_size
		self.mid_emb = emb_size // 2
		self.top_emb = emb_size // 4
		self.quantizer_top = Quantization(dict_len, self.top_emb)
		self.quantizer_middle = Quantization(dict_len, self.mid_emb)
		self.quantizer_bottom = Quantization(dict_len, self.emb_size)
		self.fusion = ConditionalFusion()
		self.classifier = Classifier(emb_size, num_classes)

	def forward(self, eeg, nirs):
		eeg_token = self.eeg_conv(eeg) # [16, 8, 1, 1] = [B, num_DWConv, 1, 1]
		nirs_token = self.nirs_conv(nirs) # [16, 8, 1, 1]
		eeg_token = eeg_token.squeeze()
		eeg_token = self.interpolation(eeg_token)
		nirs_token = nirs_token.squeeze()
		nirs_token = self.interpolation(nirs_token) # [B, E]

		eeg_top = self.pooling(eeg_token, 4)
		nirs_top = self.pooling(nirs_token, 4)
		quan_eeg_top, quan_nirs_top, quan_loss_top = self.quantizer_top(eeg_top, nirs_top)

		eeg_middle = self.pooling(eeg_token, 2)
		nirs_middle = self.pooling(nirs_token, 2)
		eeg_middle = self.fusion(quan_eeg_top, eeg_middle)
		nirs_middle = self.fusion(quan_nirs_top, nirs_middle)
		quan_eeg_middle, quan_nirs_middle, quan_loss_middle = self.quantizer_middle(eeg_middle, nirs_middle)

		eeg_bottom = self.fusion(quan_eeg_middle, eeg_token)
		nirs_bottom = self.fusion(quan_nirs_middle, nirs_token)
		quan_eeg_bottom, quan_nirs_bottom, quan_loss_bottom = self.quantizer_bottom(eeg_bottom, nirs_bottom)
		outputs = self.classifier(quan_eeg_middle, quan_nirs_middle, quan_eeg_bottom, quan_nirs_bottom)

		return {
			'outputs': outputs,
			'quan_loss': quan_loss_top + quan_loss_middle + quan_loss_bottom
		}