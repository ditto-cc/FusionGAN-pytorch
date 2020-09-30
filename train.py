# coding: utf-8
import os

import torch

from model import G, D
from utils import read_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CGAN:
    def __init__(self, config: dict):
        self.config = config

        self.G = G().to(device)
        self.D = D().to(device)

    def train(self):

        if self.config['is_train']:
            data_dir_ir = os.path.join('./{}'.format(self.config['ckpt_dir']), 'Train_ir', 'train.h5')
            data_dir_vi = os.path.join('./{}'.format(self.config['ckpt_dir']), 'Train_vi', 'train.h5')
        else:
            data_dir_ir = os.path.join('./{}'.format(self.config['ckpt_dir']), 'Test_ir', 'test.h5')
            data_dir_vi = os.path.join('./{}'.format(self.config['ckpt_dir']), 'Test_vi', 'test.h5')

        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)

        batch_size = self.config['batch_size']
        if self.config['is_train']:
            batch_idxs = len(train_data_ir) // batch_size
            for idx in range(batch_idxs):
                start_idx = idx * batch_size
                batch_images_ir = train_data_ir[start_idx: start_idx + batch_size]
                batch_labels_ir = train_data_ir[start_idx: start_idx + batch_size]
                batch_images_vi = train_data_vi[start_idx: start_idx + batch_size]
                batch_labels_vi = train_data_vi[start_idx: start_idx + batch_size]

                for _ in range(2):
                    # todo train D
                    pass
                # todo train G
                # todo add summary
                # todo print log

        pass
