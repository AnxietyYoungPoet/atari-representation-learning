import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from .utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF


class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels

        self.logvar_fc  = nn.Linear(in_features=self.final_conv_size, out_features=self.feature_size)
    
    def forward(self, x):
        f_map = self.encoder(x, fmaps=True)
        mu = f_map['out']
        logvar = self.logvar_fc(self.encoder.main[:-1](x))
        std = F.softplus(logvar)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = mu
        return z, dist, f_map


class IBSTDIMTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.vcoder = Encoder(encoder).to(device)
        self.config = config
        self.patience = self.config["patience"]
        self.beta = self.config['beta']
        # classifier1 is the global predictor
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        # classifier2 is the local predictor
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.vcoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

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
        epoch_global_loss, epoch_local_loss, epoch_kl_loss = 0., 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev in data_generator:
            z_t, dist, f_t_map = self.vcoder(x_t)
            z_tprev, _, f_tprev_map = self.vcoder(x_tprev)
            prior_dist = Normal(
                torch.zeros_like(z_t).to(self.device), torch.ones_like(z_t).to(self.device))
            kl_loss = self.beta * kl_divergence(dist, prior_dist).sum(1).mean()

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = z_t, f_tprev_map['f5']
            sy = f_t_prev.size(1)
            sx = f_t_prev.size(2)

            N = f_t.size(0)
            global_loss = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier1(f_t)
                    positive = f_t_prev[:, y, x, :]
                    # logits: N x N matrix with the diag be the postive scores
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    global_loss += step_loss
            global_loss = global_loss / (sx * sy)

            # Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = f_t_map['f5']
            local_loss = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier2(f_t[:, y, x, :])
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    local_loss += step_loss
            local_loss = local_loss / (sx * sy)
            loss = local_loss + global_loss + kl_loss

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_global_loss += global_loss.detach().item()
            epoch_local_loss += local_loss.detach().item()
            epoch_kl_loss += kl_loss.detach().item()
            steps += 1
        self.log_results(
            epoch, epoch_global_loss / steps, epoch_local_loss / steps,
            epoch_kl_loss / steps, epoch_loss / steps, prefix=mode)
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

    def log_results(
        self, epoch_idx, epoch_global_loss, epoch_local_loss,
        epoch_kl_loss, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss,
                        prefix + '_global_loss': epoch_global_loss,
                        prefix + '_local_loss': epoch_local_loss,
                        prefix + '_kl_loss': epoch_kl_loss}, step=epoch_idx, commit=False)
