# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules import conv


def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    # xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv_block1 = nn.Sequential(
            SNConv2d(2, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))

        self.conv_block2 = nn.Sequential(
            SNConv2d(256, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))

        self.conv_block3 = nn.Sequential(
            SNConv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))

        self.conv_block4 = nn.Sequential(
            SNConv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))

        self.conv_block5 = nn.Sequential(
            SNConv2d(32, 1, kernel_size=1, stride=1),
            nn.Tanh())

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, inf, vis):
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
            SNConv2d(1, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2))
        self.conv_block2 = nn.Sequential(
            SNConv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))
        self.conv_block3 = nn.Sequential(
            SNConv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))
        self.conv_block4 = nn.Sequential(
            SNConv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.2))
        self.linear = nn.Linear(6 * 6 * 256, 1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)
