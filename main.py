# coding: utf-8
import argparse
import os
from datetime import datetime

from train import CGAN


def main(config):
    model = CGAN(config)
    model.train()


parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--output', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--data', default='data', help='folder for dataset folder')
parser.add_argument('--logs', default='logs', help='folder for logs')
args = parser.parse_args()

if __name__ == '__main__':
    config = dict(
        epoch=10,
        batch_size=32,
        image_size=132,
        label_size=120,
        lr=1e-4,
        c_dim=1,
        scale=3,
        stride=14,
        output=args.output,
        sample_dir='sample',
        data=args.data,
        summary_dir=os.path.join(args.logs, datetime.now().strftime('%b%d_%H-%M-%S')),
        is_train=True,
        epsilon=5.0,
        lda=100,
    )

    main(config)
