# coding: utf-8

import torch
from torch import nn


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(2, 256, kernel_size=5, stride=1))
        self.bn1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=5, stride=1))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1))

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, inf, vis):
        x = torch.cat([vis, inf], dim=1)
        x = self.leaky_relu(self.bn1(self.sn_conv1(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn4(self.sn_conv4(x)))
        return torch.tanh(self.sn_conv5(x))


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=3, stride=2))

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.linear = nn.Linear(6 * 6 * 256, 1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        x = self.leaky_relu(self.sn_conv1(x))
        x = self.leaky_relu(self.bn1(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv4(x)))
        x = x.flatten(start_dim=1)
        return self.linear(x)
