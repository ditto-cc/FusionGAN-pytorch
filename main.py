# coding: utf-8
from train import CGAN


def main(config):
    model = CGAN(config)
    model.train()


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
        ckpt_dir='ckpt',
        sample_dir='sample',
        summary_dir='summary',
        is_train=True,

        epsilon=5.0,
        lda=100,
    )

    main(config)
