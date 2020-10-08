# coding: utf-8
import os
import torch

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

        self.gen_op = torch.optim.Adam(self.gen.parameters(), lr=config['lr'])
        self.dis_op = torch.optim.Adam(self.dis.parameters(), lr=config['lr'])

        self.lda = config['lda']
        self.epsilon = config['epsilon']

    def train_step(self, vis_img, inf_img, vis_y, inf_y, k=2):
        self.gen.train()
        d_loss_val = 0
        g_loss_val = 0

        fusion = self.gen(inf_img, vis_img)
        with torch.no_grad():
            fusion_detach = fusion

        for _ in range(k):
            self.dis_op.zero_grad()
            vis_output = self.dis(vis_y)
            fus_output = self.dis(fusion_detach)
            dis_loss = self.dis_loss_func(vis_output, fus_output)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis_op.step()

        self.gen_op.zero_grad()
        fus_output = self.dis(fusion)
        g_loss, v_gan_loss, content_loss = self.gen_loss_func(fus_output, fusion, vis_y, inf_y)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        self.gen_op.step()
        return d_loss_val / k, g_loss_val, v_gan_loss, content_loss

    @staticmethod
    def dis_loss_func(vis_output, fusion_output):
        return torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).to(device))) + \
               torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).to(device)))

    def gen_loss_func(self, fusion_output, fusion_img, vis_img, inf_img):
        gan_loss = torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0.7, 1.2).to(device)))
        content_loss = torch.mean(torch.square(fusion_img - inf_img)) + \
                       self.epsilon * torch.mean(torch.square(utils.gradient(fusion_img) - utils.gradient(vis_img)))
        return gan_loss + self.lda * content_loss, gan_loss, self.lda * content_loss

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

                batch_steps = len(train_data_ir) // batch_size
                epochs = self.config['epoch']
                for epoch in range(1, 1 + epochs):
                    d_loss_mean = 0
                    g_loss_mean = 0
                    content_loss_mean = 0
                    for step in range(1, 1 + batch_steps):
                        start_idx = (step - 1) * batch_size
                        inf_x = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        inf_y = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vis_x = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vis_y = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                        inf_x = torch.tensor(inf_x).float().to(device)
                        inf_y = torch.tensor(inf_y).float().to(device)
                        vis_x = torch.tensor(vis_x).float().to(device)
                        vis_y = torch.tensor(vis_y).float().to(device)

                        d_loss, g_loss, v_gan_loss, content_loss = self.train_step(vis_x, inf_x, vis_y, inf_y, 2)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss
                        content_loss_mean += content_loss
                        print('Epoch {}/{}, Step {}/{}, gen loss = {:.4f}, v_gan_loss = {:.4f}, '
                              'content_loss {:.4f}, dis loss = {:.4f}'.format(epoch, epochs, step, batch_steps,
                                                                              g_loss, v_gan_loss, content_loss, d_loss))
                    test_all(self.gen, os.path.join(self.config['output'], 'test{}'.format(epoch)))

                    d_loss_mean /= batch_steps
                    g_loss_mean /= batch_steps
                    content_loss_mean /= batch_steps
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch)
                    writer.add_scalar('scalar/content_loss', content_loss_mean, epoch)

                    # for name, param in self.gen.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('gen/'+name, param, epoch)
                    #
                    # for name, param in self.dis.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('dis/'+name, param, epoch)
            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/final_generator.pth' % (self.config['output']))
            print("Training Finished, Total EPOCH = %d" % self.config['epoch'])
