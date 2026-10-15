import torch
import torch.nn as nn


# Depthwise Separable Convolution
class DWSConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(DWSConv, self).__init__()
		self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels)
		self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

	def forward(self, input):
		x = self.depth_conv(input)
		x = self.point_conv(x)
		return x

class Encoder_Common(nn.Module):
	def __init__(self, num_class, emb_size, T_Width, S_Height, num_TConv=4, num_SConv=8):
		super(Encoder_Common, self).__init__()
		# Temporal Convolution
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=num_TConv, kernel_size=(1, T_Width), stride=1, padding=0)
		self.bn1 = torch.nn.BatchNorm2d(num_TConv)

		# Spatial Convolution
		self.conv2 = DWSConv(in_channels=num_TConv, out_channels=num_SConv, kernel_size=(S_Height, 1))
		self.bn2 = torch.nn.BatchNorm2d(num_SConv)

		self.fc = torch.nn.Linear(num_SConv, num_class)
		self.act = torch.nn.Sigmoid()
		self.interpolation = torch.nn.Linear(num_SConv, emb_size)

	def forward(self, x):
		x = self.act(self.bn1(self.conv1(x)))
		x = self.act(self.bn2(self.conv2(x)))
		x = x.squeeze()
		x = self.interpolation(x)
		return x