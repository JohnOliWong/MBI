import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        # data.shape = [60, 1, 500, 30]
        self.T = 500 # timepoints
        self.num_classes = 2
        
        # Layer 1
        # [60, 1, 500, 30] > [60, 16, 500, 1] > [60, 1, 16, 500]
        self.conv1 = nn.Conv2d(1, 16, (1, 30), padding = 0) # input = 1, output = 16, kernel_size = (1, 30) -> spatial convolution
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        # [60, 1, 16, 500] > [60, 4, 8, 125]
        # padding [(left, right,) (top, bottom] = [(time), (space)]
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1)) # [60, 1, 17, 533]
        self.conv2 = nn.Conv2d(1, 4, (2, 34)) # [60, 4, 16, 500]
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 4))
        
        # Layer 3
        # [60, 4, 8, 125] > [60, 4, 4, 25]
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3)) # [60, 4, 15, 128]
        self.conv3 = nn.Conv2d(4, 4, (8, 4)) # [60, 4, 8, 125]
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 5))
        
        # FC Layer
        # this dimension will depend on the number of timestamps per trial in the dataset
        self.fc1 = nn.Linear(4*4*25, self.num_classes)
    
    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.contiguous()
        x = x.view(-1, 4*4*25) # [16, 400]
        x = F.sigmoid(self.fc1(x))
        return x