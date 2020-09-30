# coding: utf-8

import torch
from torch import nn


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Tanh())

    def forward(self, vis, inf):
        inputs = torch.cat([vis, inf], dim=1)
        inputs = self.conv_block1(inputs)
        inputs = self.conv_block2(inputs)
        inputs = self.conv_block3(inputs)
        inputs = self.conv_block4(inputs)
        return self.conv_block5(inputs)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.LeakyReLU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        self.linear = nn.Linear(6*6*256, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return self.linear(x)



