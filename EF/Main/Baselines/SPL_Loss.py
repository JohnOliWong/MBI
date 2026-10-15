import torch
import torch.nn as nn
from torch import Tensor


class SPL_Loss(nn.CrossEntropyLoss):
	'''
	args:
		batch_size: batch size
		alpha：coefficient for confidence loss
		beta： coefficient for KL-divergence loss
		spl_lambda：threshold
		threshold deduce factor
	'''
	def __init__(self, *args, batch_size, alpha, beta, spl_lambda, spl_gamma, **kwargs):
		super(SPL_Loss, self).__init__(*args, **kwargs)
		self.alpha = alpha
		self.beta = beta
		self.spl_lambda = spl_lambda
		self.spl_gamma = spl_gamma

	def forward(self, input: Tensor, target: Tensor, index: Tensor, c_loss: Tensor, kd_loss: Tensor) -> Tensor:
		# find the total loss function of the current batch
		super_loss = nn.functional.cross_entropy(input, target, reduction='none') + self.alpha * c_loss + self.beta * kd_loss # [B]
		
		# determine if the current batch belongs to simple or hard sample
		v = self.spl_loss(super_loss) # self.v.shape = [num_batch, B] v.shape = [B]
		return (super_loss * v).mean()

	def increase_threshold(self):
		# increase the threshold so that more samples will be categorized as simple
		self.spl_lambda *= self.spl_gamma

	def spl_loss(self, super_loss):
		# if total loss is smaller than the threshold, it's simple sample
		v = super_loss < self.spl_lambda
		return v.int()