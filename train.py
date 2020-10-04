# coding: utf-8
import os

import torch
import tqdm

import preprocessing
import utils
from tensorboardX import SummaryWriter

from model import G, D
from test import test_all

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CGAN:
    def __init__(self, config: dict):
        self.config = config

        self.gen = G().to(device)
        self.dis = D().to(device)

        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=config['lr'])
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=config['lr'])

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, vis_img, inf_img, vis_y, inf_y, k=2):
        self.gen.train()
        d_loss_val = 0
        g_loss_val = 0

        self.set_requires_grad(self.dis, True)
        self.set_requires_grad(self.gen, False)
        fusion = self.gen(inf_img, vis_img)
        for _ in range(k):
            self.dis_optim.zero_grad()
            vis_output = self.dis(vis_y)
            fus_output = self.dis(fusion)
            dis_loss = self.dis_loss_func(vis_output, fus_output)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward()
            self.dis_optim.step()

        self.set_requires_grad(self.dis, False)
        self.set_requires_grad(self.gen, True)
        fusion = self.gen(inf_img, vis_img)
        self.gen_optim.zero_grad()
        fus_output = self.dis(fusion)
        g_loss = self.gen_loss_func(fus_output, fusion, vis_y, inf_y)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward()
        self.gen_optim.step()
        return d_loss_val / k, g_loss_val

    @staticmethod
    def dis_loss_func(vis_output, fusion_output):
        return torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).to(device))) + \
               torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).to(device)))

    def gen_loss_func(self, fusion_output, fusion_img, vis_img, inf_img):
        v_gan_loss = torch.mean(
            torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0.7, 1.2).to(device)))
        content_loss = torch.mean(torch.square(fusion_img - inf_img)) + \
                       self.config['epsilon'] * torch.mean(
            torch.square(utils.gradient(fusion_img) - utils.gradient(vis_img)))
        return v_gan_loss + self.config['lda'] * content_loss

    def train(self):
        if self.config['is_train']:
            data_dir_ir = os.path.join(self.config['data'], 'Train_ir')
            data_dir_vi = os.path.join(self.config['data'], 'Train_vi')
        else:
            data_dir_ir = os.path.join(self.config['data'], 'Test_ir')
            data_dir_vi = os.path.join(self.config['data'], 'Test_ir')

        train_data_ir, train_label_ir = preprocessing.get_images2(data_dir_ir, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])
        train_data_vi, train_label_vi = preprocessing.get_images2(data_dir_vi, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])

        batch_size = self.config['batch_size']

        if self.config['is_train']:
            with SummaryWriter(self.config['summary_dir']) as writer:

                batch_idxs = len(train_data_ir) // batch_size
                epochs = self.config['epoch']
                for epoch in range(epochs):
                    d_loss_mean = 0
                    g_loss_mean = 0
                    for idx in range(batch_idxs):
                        start_idx = idx * batch_size
                        inf_x = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                        inf_y = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                        vis_x = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1
                        vis_y = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2]) / 127.5 - 1

                        inf_x = torch.tensor(inf_x).float().to(device)
                        inf_y = torch.tensor(inf_y).float().to(device)
                        vis_x = torch.tensor(vis_x).float().to(device)
                        vis_y = torch.tensor(vis_y).float().to(device)

                        d_loss, g_loss = self.train_step(vis_x, inf_x, vis_y, inf_y)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss
                        print('Epoch {}/{}, Step {}/{}, gen loss = {:.4f}, dis loss = {:.4f}'.format(epoch + 1, epochs,
                                                                                                     idx + 1, batch_idxs,
                                                                                                     g_loss, d_loss))
                    test_all(self.gen, os.path.join(self.config['output'], 'test{}'.format(epoch)))

                    d_loss_mean /= batch_idxs
                    g_loss_mean /= batch_idxs
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch + 1)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch + 1)

                    # for name, param in self.gen.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('gen/'+name, param, epoch + 1)
                    #
                    # for name, param in self.dis.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('dis/'+name, param, epoch + 1)
            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/final_generator.pth' % (self.config['output']))
            print("Training Finished, Total EPOCH = %d" % self.config['epoch'])
