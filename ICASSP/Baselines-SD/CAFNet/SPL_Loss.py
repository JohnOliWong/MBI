import torch
import torch.nn as nn
from torch import Tensor

# class MySpl_loss(nn.CrossEntropyLoss):
# 	def __init__(self, *args, n_samples=0, batch_size, alpha, beta, spl_lambda, spl_gamma, **kwargs):
# 		super(MySpl_loss, self).__init__(*args, **kwargs)
# 		self.spl_lambda = spl_lambda
# 		self.spl_gamma = spl_gamma
# 		self.alpha = alpha
# 		self.beta = beta
# 		self.v = torch.zeros(n_samples, batch_size).int() # v.shape = [n_samples, batch_size]

# 	def forward(self, input: Tensor, target: Tensor, index: Tensor, c_loss: Tensor, kd_loss: Tensor) -> Tensor:
# 		super_loss = nn.functional.cross_entropy(input, target, reduction='none') + self.alpha * c_loss + self.beta * kd_loss
# 		v = self.spl_loss(super_loss)
# 		self.v[index] = v
# 		return (super_loss * v).mean()

# 	def increase_threshold(self):
# 		self.spl_lambda *= self.spl_gamma

# 	def spl_loss(self, super_loss):
# 		v = super_loss < self.spl_lambda
# 		return v.int()

class MySpl_loss(nn.CrossEntropyLoss):
    def __init__(self, *args, n_samples=0, batch_size, alpha, beta, spl_lambda, spl_gamma, **kwargs):
        super(MySpl_loss, self).__init__(*args, **kwargs)
        self.spl_lambda = spl_lambda
        self.spl_gamma = spl_gamma
        self.alpha = alpha
        self.beta = beta
        self.v = torch.zeros(n_samples).int()  # 改为 1D 掩码 [n_samples]
        self.current_batch_start = 0  # 记录当前 batch 的起始位置

    def forward(self, input: Tensor, target: Tensor, c_loss: Tensor, kd_loss: Tensor) -> Tensor:
        super_loss = nn.functional.cross_entropy(input, target, reduction='none') + self.alpha * c_loss + self.beta * kd_loss
        v = self.spl_loss(super_loss)  # 计算当前 batch 的掩码 [batch_size]
        
        # 更新 self.v 的对应位置
        batch_size = target.size(0)
        self.v[self.current_batch_start : self.current_batch_start + batch_size] = v
        self.current_batch_start += batch_size  # 移动到下一个 batch 的位置
        
        return (super_loss * v).mean()

    def increase_threshold(self):
        self.spl_lambda *= self.spl_gamma

    def spl_loss(self, super_loss):
        return (super_loss < self.spl_lambda).int()

    def reset_batch_counter(self):
        """在每个 epoch 开始时调用，重置 batch 位置计数器"""
        self.current_batch_start = 0