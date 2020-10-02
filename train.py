# coding: utf-8

import torch
import tqdm

import preprocessing
import utils
from model import G, D

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CGAN:
    def __init__(self, config: dict):
        self.config = config

        self.gen = G().to(device)
        self.dis = D().to(device)

        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=config['lr'])
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=config['lr'])

    def train_step(self, vis_img, inf_img, vis_target, inf_target, k=2):
        self.dis.train()
        self.gen.train()
        d_loss_val = 0
        g_loss_val = 0
        fusion_img = self.gen(vis_img, inf_img)
        for _ in range(k):
            self.dis_optim.zero_grad()
            vis_output = self.dis(vis_target)
            fus_output = self.dis(fusion_img.detach())
            dis_loss = self.dis_loss_func(vis_output, fus_output)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis_optim.step()

        self.gen_optim.zero_grad()
        fus_output = self.dis(fusion_img)
        g_loss = self.gen_loss_func(fus_output) + self.config['lda'] * self.content_loss(fusion_img, vis_target, inf_target)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        self.gen_optim.step()
        return d_loss_val / k, g_loss_val

    @staticmethod
    def dis_loss_func(vis_output, fusion_output):
        return torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).to(device))) + \
               torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).to(device)))

    def content_loss(self, fusion_img, vis_img, inf_img):
        return torch.mean(torch.square(fusion_img - inf_img) +
                          self.config['epsilon'] * torch.square(utils.gradient(fusion_img) - utils.gradient(vis_img)))

    @staticmethod
    def gen_loss_func(fusion_output):
        return torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0.7, 1.2).to(device)))

    def train(self):

        if self.config['is_train']:
            data_dir_ir = 'Train_ir'
            data_dir_vi = 'Train_vi'
        else:
            data_dir_ir = 'Test_ir'
            data_dir_vi = 'Test_vi'

        train_data_ir, train_label_ir = preprocessing.get_images2(data_dir_ir, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])
        train_data_vi, train_label_vi = preprocessing.get_images2(data_dir_vi, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])

        batch_size = self.config['batch_size']

        if self.config['is_train']:
            batch_idxs = len(train_data_ir) // batch_size
            epochs = self.config['epoch']
            for epoch in range(epochs):
                d_loss_mean = 0
                g_loss_mean = 0
                desc = 'Epoch {}/{}, Step'.format(epoch + 1, epochs)
                for idx in tqdm.trange(batch_idxs, desc=desc, total=batch_idxs):
                    start_idx = idx * batch_size
                    inf_img = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                    inf_target = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                    vis_img = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                    vis_target = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1

                    inf_img = torch.tensor(inf_img).float().to(device)
                    inf_target = torch.tensor(inf_target).float().to(device)
                    vis_img = torch.tensor(vis_img).float().to(device)
                    vis_target = torch.tensor(vis_target).float().to(device)

                    d_loss, g_loss = self.train_step(vis_img, inf_img, vis_target, inf_target)
                    d_loss_mean += d_loss
                    g_loss_mean += g_loss
                d_loss_mean /= batch_idxs
                g_loss_mean /= batch_idxs
                print('Epoch {}/{}, gen loss = {:.4f}, dis loss = {:.4f}'.format(epoch + 1, epochs, g_loss_mean,
                                                                                 d_loss_mean))
