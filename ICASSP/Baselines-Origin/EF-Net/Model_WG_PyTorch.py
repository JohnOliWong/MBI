import torch
import torch.nn as nn
import torch.nn.functional as F

#EF-Net
'''
data [trial, time, channel, 1]
'''

class EF_Net(nn.Module):
	def __init__(self):
		super(EF_Net, self).__init__()

		self.eeg_conv1 = nn.Sequential(
		nn.Conv2d(1, 32, kernel_size=(8, 1)),
		nn.ReLU()
		)
		self.eeg_conv2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(8, 1)),
			nn.ReLU()
		)
		self.eeg_pool1 = nn.MaxPool2d(kernel_size=(8, 1))
		self.eeg_dropout1 = nn.Dropout(0.5)
		self.eeg_bn1 = nn.BatchNorm2d(32)

		self.eeg_conv4 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=(4, 4)),
			nn.ReLU()
		)
		self.eeg_conv5 = nn.Sequential(
		nn.Conv2d(64, 64, kernel_size=(4, 4)),
		nn.ReLU()
		)
		self.eeg_pool2 = nn.MaxPool2d(kernel_size=(4, 4))
		self.eeg_dropout2 = nn.Dropout(0.5)
		self.eeg_bn2 = nn.BatchNorm2d(64)

		self.eeg_fc1 = nn.Sequential(
			nn.Linear(self._get_eeg_fc_input_size(), 256),
			nn.ReLU()
		)
		self.eeg_dropout3 = nn.Dropout(0.5)
		self.eeg_fc2 = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU()
		)

		self.nirs_conv1 = nn.Sequential(
		nn.Conv2d(1, 32, kernel_size=(4, 1)),
		nn.ReLU()
		)
		self.nirs_conv2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(4, 1)),
			nn.ReLU()
		)
		self.nirs_pool1 = nn.MaxPool2d(kernel_size=(4, 1))
		self.nirs_dropout1 = nn.Dropout(0.5)
		self.nirs_bn1 = nn.BatchNorm2d(32)

		self.nirs_conv3 = nn.Sequential(
		nn.Conv2d(32, 64, kernel_size=(2, 2)),
		nn.ReLU()
		)
		self.nirs_conv4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=(2, 2)),
			nn.ReLU()
		)
		self.nirs_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
		self.nirs_dropout2 = nn.Dropout(0.5)
		self.nirs_bn2 = nn.BatchNorm2d(64)

		self.nirs_fc1 = nn.Sequential(
			nn.Linear(self._get_nirs_fc_input_size(), 128),
			nn.ReLU()
		)

		self.combined_fc1 = nn.Sequential(
			nn.Linear(256, 256),  # 128 (eeg) + 128 (fnirs) = 256
			nn.ReLU()
		)
		self.combined_dropout = nn.Dropout(0.5)
		self.combined_fc2 = nn.Sequential(
			nn.Linear(256, 64),
			nn.ReLU()
		)
		self.output_layer = nn.Sequential(
			nn.Linear(64, 2), # cross entropy loss requires the column number to be the class number
			nn.Sigmoid()
		)

	def _get_eeg_fc_input_size(self):
		x = torch.zeros(1, 2000, 30, 1)
		x = x.permute(0, 3, 1, 2)
		x = self.eeg_conv1(x)
		x = self.eeg_conv2(x)
		x = self.eeg_conv2(x)
		x = self.eeg_pool1(x)
		x = self.eeg_conv4(x)
		x = self.eeg_conv5(x)
		x = self.eeg_conv5(x)
		x = self.eeg_pool2(x)
		return x.contiguous().view(1, -1).size(1)
	
	def _get_nirs_fc_input_size(self):
		x = torch.zeros(1, 100, 72, 1)
		x = x.permute(0, 3, 1, 2)
		x = self.nirs_conv1(x)
		x = self.nirs_conv2(x)
		x = self.nirs_pool1(x)
		x = self.nirs_conv3(x)
		x = self.nirs_conv4(x)
		x = self.nirs_pool2(x)
		return x.contiguous().view(1, -1).size(1)
	
	def forward(self, eeg, nirs):
		# EEG pathway
		eeg = eeg.permute(0, 3, 1, 2)
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
		e = e.contiguous().view(e.size(0), -1)  # Flatten

		e = self.eeg_fc1(e)
		e = self.eeg_dropout3(e)
		e = self.eeg_fc2(e)
		
		# fNIRS pathway
		nirs = nirs.permute(0, 3, 1, 2)
		f = self.nirs_conv1(nirs)
		f = self.nirs_conv2(f)
		f = self.nirs_pool1(f)
		f = self.nirs_dropout1(f)
		f = self.nirs_bn1(f)
		
		f = self.nirs_conv3(f)
		f = self.nirs_conv4(f)
		f = self.nirs_pool2(f)
		f = self.nirs_dropout2(f)
		f = self.nirs_bn2(f)
		f = f.contiguous().view(f.size(0), -1)  # Flatten

		f = self.nirs_fc1(f)
		
		# Combined pathway
		combined = torch.cat((e, f), dim=1)
		z = self.combined_fc1(combined)
		z = self.combined_dropout(z)
		z = F.normalize(z, p=2, dim=1)  # L2 normalization
		z = self.combined_fc2(z)
		output = self.output_layer(z)
		
		return output

def ef_net():
	return EF_Net()