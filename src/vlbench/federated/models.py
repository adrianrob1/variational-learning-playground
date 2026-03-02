# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LinearModel(nn.Module):
    def __init__(self, D_in, D_out, bias=False):
        super(LinearModel, self).__init__()
        self.upper = nn.Linear(D_in, D_out, bias=bias)
        init.zeros_(self.upper.weight)
        if bias:
            init.zeros_(self.upper.bias)

    def forward(self, x):
        return self.upper(x)


class CifarNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CifarNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512), nn.ReLU(), nn.Dropout(p=0.5)
        )
        self.upper = nn.Linear(512, out_channels)

    def forward(self, x):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.upper(o)
        return o


class FedDynCifarCNN(nn.Module):
    def __init__(self, n_cls=10, in_channels=3):
        super(FedDynCifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FilterResponseNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__()
        shape = (1, num_features, 1, 1)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.ones(*shape))
        self.beta = nn.Parameter(torch.zeros(*shape))
        self.tau = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        avg_dims = (2, 3)
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalisation="GroupNorm"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(2, planes)
            if normalisation == "GroupNorm"
            else FilterResponseNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(2, planes)
            if normalisation == "GroupNorm"
            else FilterResponseNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, normalisation="GroupNorm", num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.normalisation = normalisation
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = (
            nn.GroupNorm(2, 16)
            if self.normalisation == "GroupNorm"
            else FilterResponseNorm2d(16)
        )
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, normalisation=self.normalisation)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.normalisation == "GroupNorm":
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
        else:
            out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, normalisation="GroupNorm"):
    return ResNet(
        BasicBlock, [3, 3, 3], num_classes=num_classes, normalisation=normalisation
    )
