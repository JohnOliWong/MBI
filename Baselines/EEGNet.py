import torch
import torch.nn as nn
import torch.nn.functional as F


# 26.09.23改进：
# 1. fc1计算输入维度，从T // 32改为T // 4 // 8
# 2. 删除末尾多余的softmax

class EEGNet(nn.Module):
    def __init__(self, C, T, num_classes, f1, depth, f2):
        super(EEGNet, self).__init__()
        self.C = C
        self.T = T
        self.num_classes = num_classes
        self.f1 = f1
        self.depth = depth
        self.f2 = f2

        # Layer 1: Conv2D
        # [1, C, T] -> [F1, C, T] same padding
        self.conv1 = nn.Conv2d(1, self.f1, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(self.f1)

        # Layer 2: DepthwiseConv2D
        # [F1, C, T] -> [D * F1, 1, T] -> [D * F1, 1, T // 4]
        self.depthwise_conv2 = nn.Conv2d(self.f1, self.f1 * self.depth, (self.C, 1), padding=0, groups=self.f1)
        self.batchnorm2 = nn.BatchNorm2d(self.f1 * self.depth)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 4))
        self.dropout2 = nn.Dropout(0.25)

        # Layer 3: SeparableConv2D
        # [D * F1, 1, T // 4] -> [F2, 1, T // 4] -> [F2, 1, T // 32]
        self.separable_conv3 = nn.Sequential(
            nn.Conv2d(self.f1 * self.depth, self.f1 * self.depth, (1, 16), padding='same', groups=self.f1 * self.depth),
            nn.Conv2d(self.f1 * self.depth, self.f2, (1, 1))
        )
        self.batchnorm3 = nn.BatchNorm2d(self.f2)
        self.activation3 = nn.ELU()
        self.avg_pool3 = nn.AvgPool2d((1, 8))
        self.dropout3 = nn.Dropout(0.25)

        # Classfication Head
        # feature map time length after two pooling stages: T // 4 then // 8
        self.fc1 = nn.Linear(self.f2 * (self.T // 4 // 8), self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)

        x = self.depthwise_conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.separable_conv3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.avg_pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x