import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from .utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF
from a2c_ppo_acktr.utils import init


class MaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        last_init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 5),
            nn.init.calculate_gain('relu'))
        
        self.main = nn.Sequnetial(
            init_(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            last_init_(nn.Conv2d(64, 128, 4, stride=2)),
            nn.Sigmoid(),
        )
        self.train()
    
    def forward(self, inputs):
        return self.main(inputs)


class InfoBottleneckNoiseTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        # classifier1 is the global predictor
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        # classifier2 is the local predictor
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth).to(device)
        self.mask = MaskEncoder()
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
            list(self.classifier2.parameters()) + list(self.mask.parameters()),
            lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                x_t.append(episode[t])

                x_tprev.append(episode[t - 1])
                ts.append([t])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev in data_generator:
            f_t_f5, f_t_prev_f5 = self.encoder.main[:6](x_t), self.encoder.main[:6](x_tprev)
            f5 = torch.stack([f_t_f5, f_t_prev_f5], dim=0)
            mu = f5.mean(0).detach()
            std = f5.std(0).detach()
            alpha_t, alpha_tprev = self.mask(x_t), self.mask(x_tprev)
            f_t_f5_norm = (f_t_f5 - mu) / (std + 1e-6)
            f_t_prev_f5_norm = (f_t_prev_f5 - mu) / (std + 1e-6)

            z_mu_t = alpha_t * f_t_f5_norm
            z_mu_tprev = alpha_tprev * f_t_prev_f5_norm
            eps_t = mu.data.new(z_mu_t.size()).normal_()
            eps_tprev = mu.data.new(z_mu_t.size()).normal_()
            f_t = self.encoder.main[6:](z_mu_t)
            f_t_prev = z_mu_tprev.permute(0, 2, 3, 1)


            # Loss 1: Global at time t, f5 patches at time t-1
            sy = f_t_prev.size(1)
            sx = f_t_prev.size(2)

            N = f_t.size(0)
            loss1 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier1(f_t)
                    positive = f_t_prev[:, y, x, :]
                    # logits: N x N matrix with the diag be the postive scores
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    loss1 += step_loss
            loss1 = loss1 / (sx * sy)

            # Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = z_mu_t.permute(0, 2, 3, 1)
            loss2 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier2(f_t[:, y, x, :])
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    loss2 += step_loss
            loss2 = loss2 / (sx * sy)
            loss = loss1 + loss2

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()
            #preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            #accuracy1 += calculate_accuracy(preds1, target)
            #preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            #accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train(), self.classifier2.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.classifier1.eval(), self.classifier2.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss2, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss,
                        prefix + '_loss1': epoch_loss1,
                        prefix + '_loss2': epoch_loss2}, step=epoch_idx, commit=False)
