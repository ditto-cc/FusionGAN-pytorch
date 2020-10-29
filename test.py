# coding: utf-8
import imageio
import torch

import os

import glob
import time

from model import G
import cv2
from metric import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data2(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data


def input_setup2(data_vi, data_ir, index):
    padding = 6
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread(data_ir[index])
    _vi = imread(data_vi[index])
    # input_ir = (_ir - 127.5) / 127.5
    input_ir = _ir / 255
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    # input_vi = (_vi - 127.5) / 127.5
    input_vi = _vi / 255
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :, 0]


def imsave(image, path):
    return imageio.imwrite(path, image)


def test_all(g=None, path=os.path.join(os.getcwd(), 'output', 'result'), data='data'):
    data_ir = prepare_data2(os.path.join(data, 'Test_ir'))
    data_vi = prepare_data2(os.path.join(data, 'Test_vi'))

    if g is None:
        g = G().to(device)
        weights = torch.load('output/final_generator.pth')
        g.load_state_dict(weights)

    if not os.path.exists(path):
        os.makedirs(path)

    g.eval()
    with torch.no_grad():
        for i in range(len(data_ir)):
            start = time.time()
            train_data_ir, train_data_vi = input_setup2(data_vi, data_ir, i)
            train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
            train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

            train_data_ir = torch.tensor(train_data_ir).float().to(device)
            train_data_vi = torch.tensor(train_data_vi).float().to(device)

            result = g(train_data_ir, train_data_vi)
            result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(result)
            save_path = os.path.join(path, str(i + 1) + ".bmp")
            end = time.time()
            #
            imsave(result, save_path)
            # print("Testing [%d] success,Testing time is [%f]" % (i, end - start))
            pass


if __name__ == '__main__':
    test_all()
