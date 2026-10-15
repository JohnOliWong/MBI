import torch
import torch.nn as nn
import torch.nn.functional as F


# 26.09.24改进：
# 删除分类头结尾的Sigmoid

class EF_Net(nn.Module):
	def __init__(self, num_classes, mode, eeg_time=None):
		super(EF_Net, self).__init__()
		self.num_classes = num_classes
		self.channels = [0, 0]
		
		# 各数据集的输入维度（eye 分支已改为时间无关，任意 K 均可；eeg 时间由外部传入）
		if mode in ('SEEDIV', 'SEEDV'):
			self.eeg_channels = 62
			self.eeg_time = 4000 if eeg_time is None else eeg_time
			self.eye_channels = 31 if mode == 'SEEDIV' else 33   # SEED-IV 31 维 / SEED-V 33 维
		elif mode in ('EFMI', 'EFMA'):
			self.eeg_channels = 30
			self.eeg_time = 4000 if eeg_time is None else eeg_time
			self.eye_channels = 72
		elif mode == 'EFWG':
			self.eeg_channels = 30
			self.eeg_time = 2000 if eeg_time is None else eeg_time
			self.eye_channels = 72
		
		# eeg block 1
		self.eeg_conv1 = nn.Sequential(
		nn.Conv2d(1, 8, kernel_size=(1, 16)),
		nn.ReLU()
		)
		self.eeg_conv2 = nn.Sequential(
			nn.Conv2d(8, 8, kernel_size=(1, 16)),
			nn.ReLU()
		)
		self.eeg_pool1 = nn.MaxPool2d(kernel_size=(1, 16))
		self.eeg_dropout1 = nn.Dropout(0.5)
		self.eeg_bn1 = nn.BatchNorm2d(8)

		# eeg block 2
		self.eeg_conv4 = nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=(8, 8)),
			nn.ReLU()
		)
		self.eeg_conv5 = nn.Sequential(
		nn.Conv2d(16, 16, kernel_size=(8, 8)),
		nn.ReLU()
		)
		self.eeg_pool2 = nn.MaxPool2d(kernel_size=(8, 8))
		self.eeg_dropout2 = nn.Dropout(0.5)
		self.eeg_bn2 = nn.BatchNorm2d(16)

		# eeg block 3
		x = torch.randn(1, 1, self.eeg_channels, self.eeg_time)
		x = self.eeg_conv1(x)
		x = self.eeg_conv2(x)
		x = self.eeg_conv2(x)
		x = self.eeg_pool1(x)
		x = self.eeg_conv4(x)
		x = self.eeg_conv5(x)
		x = self.eeg_conv5(x)
		x = self.eeg_pool2(x)
		self.channels[0] = x.contiguous().view(1, -1).size(1)
		self.eeg_fc1 = nn.Sequential(
			nn.Linear(self.channels[0], 256),
			nn.ReLU()
		)
		self.eeg_dropout3 = nn.Dropout(0.5)
		self.eeg_fc2 = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU()
		)

		# eye 分支（时间无关：通道卷到 8→16，再用 AdaptiveAvgPool 把任意 K 压到 1）
		self.eye_conv1 = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=(self.eye_channels, 1), bias=False),
			nn.BatchNorm2d(8),
			nn.ReLU()
		)
		self.eye_conv2 = nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1), bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU()
		)
		self.eye_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.eye_dropout = nn.Dropout(0.5)
		self.eye_fc1 = nn.Sequential(
			nn.Linear(16, 128),
			nn.ReLU()
		)

		# fusion
		self.combined_fc1 = nn.Sequential(
			nn.Linear(256, 256), # 128 (eeg) + 128 (eye) = 256
			nn.ReLU()
		)
		self.combined_dropout = nn.Dropout(0.5)
		self.combined_fc2 = nn.Sequential(
			nn.Linear(256, 64),
			nn.ReLU()
		)
		self.output_layer = nn.Sequential(
			nn.Linear(64, self.num_classes), # cross entropy loss requires the column number to be the class number
		)

	def forward(self, eeg, eye):
		# EEG pathway
		e = self.eeg_conv1(eeg)
		e = self.eeg_conv2(e)
		e = self.eeg_conv2(e)
		e = self.eeg_pool1(e)
		e = self.eeg_dropout1(e)
		e = self.eeg_bn1(e)
		
		e = self.eeg_conv4(e)
		e = self.eeg_conv5(e)
		e = self.eeg_conv5(e)
		e = self.eeg_pool2(e)
		e = self.eeg_dropout2(e)
		e = self.eeg_bn2(e)
		e = e.contiguous().view(e.size(0), -1)

		e = self.eeg_fc1(e)
		e = self.eeg_dropout3(e)
		e = self.eeg_fc2(e)
		
		# eye pathway
		f = self.eye_conv1(eye)            # [B, 8, 1, K]
		f = self.eye_conv2(f)              # [B, 16, 1, K]
		f = self.eye_pool(f)               # [B, 16, 1, 1]
		f = self.eye_dropout(f)
		f = f.contiguous().view(f.size(0), -1)   # [B, 16]

		f = self.eye_fc1(f)                # [B, 128]
		
		# Combined pathway
		combined = torch.cat((e, f), dim=1)
		z = self.combined_fc1(combined)
		z = self.combined_dropout(z)
		z = F.normalize(z, p=2, dim=1) # L2 normalization
		z = self.combined_fc2(z)
		output = self.output_layer(z)
		
		return output