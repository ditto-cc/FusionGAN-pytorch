# coding: utf-8

import h5py
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'

def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def gradient(x):
    with torch.no_grad():
        laplace = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)


if __name__ == '__main__':
    img = cv2.imread('lena.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255
    g = gradient(torch.tensor(gray).float().unsqueeze(0).unsqueeze(0))
    plt.imshow(g[0][0].numpy())
    plt.show()
