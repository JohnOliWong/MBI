import torch
import torch.nn.functional as F
from torch import nn
import random

from EFBook.Encoder_EF import Encoder_EF
from EFBook.Encoder_Common import Encoder_Common


class EGA(nn.Module):
	def __init__(self, emb_size, num_heads=4, drop_out=0.3):
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=drop_out, batch_first=True)

	def forward(self, x, y):
		# x is the dominant modality, e.g. EEG
		attn_output, _ = self.attn(query=x, key=y, value=y)
		return attn_output


class Quantization(nn.Module):
	def __init__(self, dict_len, emb_size, decay=0.99, quan_weight=1.0, ot_weight=5.0, threshold=60):
		super().__init__()
		self.dict_len = dict_len
		self.emb_size = emb_size
		self.decay = decay
		self.quan_weight = quan_weight
		self.ot_weight = ot_weight
		self.threshold = threshold
		self.codebook = nn.Parameter(torch.randn(dict_len, emb_size))
		self.register_buffer('N', torch.ones(dict_len))
		self.register_buffer('m', self.codebook.clone())
		self.register_buffer('status', torch.zeros(dict_len))

	def forward(self, cont_features):
		codebook_norm = F.normalize(self.codebook, dim=-1)
		distances = -cont_features @ codebook_norm.t()
		quan_idx = torch.argmin(distances, dim=-1)
		quan_token = self.codebook.data[quan_idx]
		quan_token = cont_features + (quan_token - cont_features).detach()
		usage = None
		if self.training:
			usage = self.codebook_update(cont_features, quan_idx)
		vq_loss = F.mse_loss(cont_features.detach(), quan_token)
		commit_loss = F.mse_loss(cont_features, quan_token.detach())
		quan_loss = vq_loss + commit_loss
		ot_loss = self.ot_loss(cont_features, quan_idx)
		codebook_loss = self.quan_weight * quan_loss + self.ot_weight * ot_loss
		if usage is not None:
			return {'usage': usage, 'quan_token': quan_token, 'codebook_loss': codebook_loss}
		else:
			return {'quan_token': quan_token, 'codebook_loss': codebook_loss}

	def codebook_update(self, features, indices):
		with torch.no_grad():
			features = features.float()
			one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()
			counts = one_hot.sum(dim=[0])
			matched = (counts > 0)
			usage = torch.count_nonzero(matched)
			if matched.any():
				sum_features = torch.einsum('nc,nd->dc', features, one_hot)
				self.N[matched] = self.decay * self.N[matched] + (1-self.decay) * counts[matched]
				self.m[matched] = self.decay * self.m[matched] + (1-self.decay) * sum_features[matched, :]
				self.codebook.data[matched] = self.m[matched] / self.N[matched].unsqueeze(1)
				self.status[matched] = 0
			inactive = ~matched
			self.status[inactive] += 1
			if (self.status >= self.threshold).any():
				self.codeword_replace(features)
			return usage

	def codeword_replace(self, features):
		inactive = (self.status >= self.threshold)
		num_replace = int(inactive.sum())
		if num_replace > 0:
			num_features = features.shape[0]
			if num_features >= num_replace:
				perm = torch.randperm(num_features)
				repl = features[perm[:num_replace]].clone()
			else:
				active_idx = torch.where(self.N > 1)[0]
				repl = features.clone()
				while repl.shape[0] < num_replace and active_idx.numel() > 0:
					random_idx = random.choice(active_idx.cpu().numpy())
					repl = torch.cat([repl, self.codebook.data[random_idx].unsqueeze(0)], dim=0)
			self.codebook.data[inactive] = repl
			self.N[inactive] = 1

	def ot_loss(self, features, indices):
		one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()
		codebook_probs = one_hot.mean(dim=[0])
		target_probs = torch.ones_like(codebook_probs) / len(self.codebook)
		ot_loss = F.kl_div(input=torch.log(codebook_probs + 1e-6), target=target_probs, reduction='batchmean')
		return ot_loss


class AttentionFusion(nn.Module):
	def __init__(self, emb_size):
		super(AttentionFusion, self).__init__()
		self.weight = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
		self.softmax = nn.Softmax(dim=-1)
		self.alpha = None

	def forward(self, out):
		o = torch.cat([i @ self.weight for i in out], dim=-1)
		if o.dim() == 1:
			o = o.unsqueeze(1)
		self.alpha = self.softmax(o)
		outputs = sum([i * self.alpha[:, index].unsqueeze(1) for index, i in enumerate(out)])
		return outputs


class Classifier(nn.Module):
	def __init__(self, emb_size, num_class, num_modalities):
		super().__init__()
		self.num_modalities = num_modalities
		self.modal_weights = nn.ModuleList([AttentionFusion(emb_size) for _ in range(num_modalities)])
		self.weight_fusion = AttentionFusion(emb_size)
		self.weights = nn.Parameter(torch.Tensor([1.] * num_modalities + [0.5]), requires_grad=True)
		self.projector = nn.Sequential(
			nn.Linear(emb_size, emb_size // 2), nn.ReLU(), nn.Dropout(0.2),
			nn.Linear(emb_size // 2, num_class),
		)

	def forward(self, tokens, quan_tokens):
		# tokens / quan_tokens: list of N [B, emb]
		N = self.num_modalities
		modal_fusions = [self.modal_weights[i]([tokens[i], quan_tokens[i]]) for i in range(N)]
		sum_fusion = torch.stack(tokens + quan_tokens).sum(dim=0)
		ws = torch.softmax(self.weights, dim=0)
		outputs = sum(ws[i] * self.projector(modal_fusions[i]) for i in range(N))
		outputs = outputs + ws[N] * self.projector(sum_fusion)
		return outputs


class Cosine_Loss(nn.Module):
	def __init__(self):
		super(Cosine_Loss, self).__init__()

	def forward(self, x, y, threshold=0.2, eps=1e-6):
		dot = torch.sum(x * y, dim=1)
		norm_x = torch.norm(x, p=2, dim=1)
		norm_y = torch.norm(y, p=2, dim=1)
		cosine_sim = dot / (norm_x * norm_y + eps)
		constraint = torch.relu(torch.abs(cosine_sim) - threshold)
		return torch.mean(constraint ** 2)


class EF_Book(nn.Module):
	def __init__(self, dict_len, emb_size, threshold, num_class, mode, device, modalities=None, eeg_time=None):
		super().__init__()
		self.emb_size = emb_size
		# 模态配置（与 Encoder_EF 共享同一份配置函数）
		self.modalities, modal_chan, modal_time, modal_pool, modal_is_eeg = Encoder_EF._modal_config(mode, modalities, eeg_time)
		self.num_modalities = len(self.modalities)

		self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size,
								  emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.3,
								  self_dropout=0.3, cross_dropout=0.3, cls_dropout=0.5,
								  num_classes=num_class, mode=mode, device=device,
								  modalities=self.modalities, eeg_time=eeg_time)

		self.common_convs = nn.ModuleList([
			Encoder_Common(num_class, emb_size, T_Width=modal_time[name], S_Height=modal_chan[name])
			for name in self.modalities
		])
		self.quantizers = nn.ModuleList([Quantization(dict_len, emb_size, threshold=threshold) for _ in self.modalities])
		self.fusion_quantizer = Quantization(dict_len, emb_size, threshold=threshold)
		self.cosine_loss = Cosine_Loss()
		self.ega = EGA(emb_size)
		self.classifier = Classifier(emb_size, num_class, self.num_modalities)

	def forward(self, inputs, last_batch):
		# inputs: dict {name: tensor [B,1,chan,time]}
		N = self.num_modalities
		p_tokens = self.ef_conv(inputs)                                            # list of N [B, emb]
		s_tokens = [self.common_convs[i](inputs[name]) for i, name in enumerate(self.modalities)]

		# disentangling loss：各模态共享 token 与私有 token 的余弦约束
		disentangling_loss = sum(self.cosine_loss(s_tokens[i], p_tokens[i]) for i in range(N))

		# 各模态私有码本
		quan_private = [self.quantizers[i](p_tokens[i]) for i in range(N)]

		# 融合：非 EEG 模态的共享 token 关注 EEG 共享 token，再 concat 进共享码本
		B = p_tokens[0].shape[0]
		fused_s = [s_tokens[0]]
		for i in range(1, N):
			fused_s.append(self.ega(s_tokens[0], s_tokens[i]))
		fusion_features = torch.cat(fused_s, dim=0)                                # [N*B, emb]
		fusion_out = self.fusion_quantizer(fusion_features)
		quan_fusion = fusion_out['quan_token']
		fusion_quan_loss = fusion_out['codebook_loss']
		quan_fusion_split = [quan_fusion[i * B:(i + 1) * B] for i in range(N)]

		quan_combined = [quan_private[i]['quan_token'] + quan_fusion_split[i] for i in range(N)]

		outputs = self.classifier(p_tokens, quan_combined)
		codebook_loss = sum(quan_private[i]['codebook_loss'] for i in range(N)) + 0.5 * fusion_quan_loss

		if last_batch:
			usage = [q.get('usage') for q in quan_private]
			if usage[0] is not None:
				print(f'Codebook Usage: {self.modalities} {usage} Shared {fusion_out.get("usage")}')

		return {
			'outputs': outputs,
			'loss': 10 * disentangling_loss + 0.05 * codebook_loss,
		}
