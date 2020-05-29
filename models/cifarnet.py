"""CIFAR model for ADDA."""

import math
import torch
import torch.nn.functional as F
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.proj = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x 
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.relu1(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        if self.downsample is not None:
            x = self.bn(self.proj(x))
        x = self.relu2(x + residual)
        return x

class CIFAREncoder(nn.Module):
    """CIFAR encoder model for ADDA."""
    def __init__(self, depth=26, width=4, num_classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        
        self.restored = False

        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(CIFAREncoder, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        # self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.inplanes = 16
        self.conv1 = nn.Conv2d(channels, 16*width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = norm_layer(16 * width)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 16*width
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # stliu: change to adaptive for different scale pictures

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = True
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for _ in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class CIFARClassifier(nn.Module):
    """CIFAR classifier model for ADDA."""

    def __init__(self, width=4):
        """Init LeNet encoder."""
        super(CIFARClassifier, self).__init__()
        self.fc2 = nn.Linear(64 * width, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
