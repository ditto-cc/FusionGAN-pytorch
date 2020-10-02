# coding: utf-8

import h5py
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient(x):
    with torch.no_grad():
        laplace = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)
