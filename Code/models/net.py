import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as Data
import numpy as np
# from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as Data
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from IPython import display
__all__ = ['ConvertTo3D', 'MLP',  'DenseNet','PreActResNet', 'PreActResNet34', 'PreActResNet50', 'PreActBlock', 'PreActBottleneck']


class Identity_Block(nn.Module):
    def __init__(self, input_features, out_features):
        super(Identity_Block, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Linear(out_features, out_features), nn.BatchNorm1d(out_features))
        self.act = nn.ReLU()
    def forward(self, x):
        Y = self.net(x)
        Y = self.act(Y + x)
        return Y
    
class Dense_Block(nn.Module):
    def __init__(self, input_features, out_features):
        super(Dense_Block, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Linear(out_features, out_features), nn.BatchNorm1d(out_features))
        self.act = nn.ReLU()
        self.short_linear = nn.Linear(input_features, out_features)
        self.short_bn = nn.BatchNorm1d(out_features)
    def forward(self, x):
        Y = self.net(x)
        print('yes')
        shortcut = self.short_bn(self.short_linear(x))
        Y = self.act(shortcut + Y)
        return Y

class ResNet50Regression(nn.Module):
    def __init__(self, input_features, width, num_targets):
        super(ResNet50Regression, self).__init__()
        blks = []
        for number, i in enumerate(range(3)):
            if number ==0:
                blks.append(Dense_Block(input_features, width))
            else:
                blks.append(Dense_Block(width, width))
            blks.append(Identity_Block(width, width))
            blks.append(Identity_Block(width, width))
        self.net =  nn.Sequential(*blks)
        self.linear = nn.Linear(width, num_targets)
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x):
        Y = self.net(x)
        Y = self.linear(self.bn(Y))
        return Y
        
class ConvertTo3D(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1)
        x = x.permute(0, 2, 1)
        return x

class RandomDrop(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        assert 0 < dropout < 1
        self.dropout = dropout
    
    def forward(self, x):
   
        mask = (torch.Tensor(x.shape).to(x.device).uniform_(0, 1) > self.dropout).float()
        return mask * x


# MLP--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_features, num_hiddens, num_outputs, num_layers=7, rate=0.7, 
                drop = None):
        
        self.drop = drop
        if drop==None:
            drop = [0.0]*(num_layers-1)
        # num_layers involves input layer
        super(MLP, self).__init__()
        self.in_features = in_features
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.net = nn.Sequential()
#       RandomDrop(dropout=0.1)

        self.net.add_module('linear1', nn.Linear(self.in_features, self.num_hiddens))
        self.num_features = self.num_hiddens
        
        for i in range(2, num_layers):
            self.net.add_module(f'bn{i-1}', nn.BatchNorm1d(self.num_features))
            self.net.add_module(f'act{i-1}', nn.ReLU())
            self.net.add_module(f'dropout{i-1}', nn.Dropout(p=drop[i - 2]))
            self.net.add_module(f'linear{i}', nn.Linear(self.num_features, int(self.num_features * rate)))
            self.num_features = int(self.num_features * rate)
        
        self.net.add_module(f'bn{num_layers-1}', nn.BatchNorm1d(self.num_features))
        self.net.add_module(f'act{num_layers-1}', nn.ReLU())
        self.net.add_module(f'dropout{num_layers-1}', nn.Dropout(p=drop[num_layers - 2]))
        self.net.add_module('out', nn.Linear(self.num_features, num_outputs, bias=True))
    def forward(self, X):
        X = self.net(X)
#         idx1 = torch.argsort(X[:, 1:4], dim=1)
#         idx2 = torch.cat([torch.zeros((X.shape[0], 1), dtype=torch.long, device=X.device), idx1 + 1, idx1 + 4], dim=1)
#         X = torch.take_along_dim(X, idx2, 1)
                
        return X


#DenseNet-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm1d(in_channels), 
                        nn.ReLU(),
                        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm1d(in_channels), 
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2))
    return blk

class DenseNet(nn.Module):
    def __init__(self, n_outputs = 9):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64), 
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            self.net.add_module("DenseBlock_%d" % i, DB)
            num_channels = DB.out_channels

            if i != len(num_convs_in_dense_blocks) - 1:
                self.net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.net.add_module("BN", nn.BatchNorm1d(num_channels))
        self.net.add_module("relu", nn.ReLU())
        self.net.add_module("global_avg_pool", nn.AdaptiveMaxPool1d(1))
        self.channels = num_channels
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_channels, n_outputs)
    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

#ResNet-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_outputs = 9):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, n_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
#         out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


# NiN-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(), 
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.BatchNorm2d(out_channels),
                       nn.ReLU(),
                       nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.BatchNorm2d(out_channels),
                       nn.ReLU())
    return blk
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.b1 = nn.Sequential(
            nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            nin_block(384, 5, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool(),
        )
    def forward(self, x):
        x = self.b1(x)
        x = x.view(x.size()[0], -1)
        return x