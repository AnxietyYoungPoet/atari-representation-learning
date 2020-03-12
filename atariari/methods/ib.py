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


class Classifier(nn.Module):
  def __init__(self, num_inputs1, num_inputs2):
    super().__init__()
    self.network = nn.Bilinear(num_inputs1, num_inputs2)

  def forward(self, x1, x2):
    return self.network(x1, x2)


class Encoder(nn.Module):
  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder
    self.feature_size = self.encoder.feature_size
    self.final_conv_size = self.encoder.final_conv_size
    self.final_conv_shape = self.encoder.final_conv_shape
    self.input_channels = self.encoder.input_channels

    self.logvar_fc = nn.Linear(in_features=self.final_conv_size, out_features=self.feature_size)

  def reparametrize(self, mu, logva):
    if self.training:
      eps = torch

  def forward(self, x):
    final_conv = self.encoder.main[:-1](x)
    mu = self.encoder.main[-1](final_conv)
    logvar = self.logvar_fc(final_conv)
    std = F.softplus(logvar)
    dist = Normal(mu, std)
    if self.training:
      z = dist.rsample()
    else:
      z = mu
    return z, mu, dist


class InfoBottleneck(Trainer):
  def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
    super().__init__(encoder, wandb, device)
    self.config = config
    self.patience = self.config['patience']
    self.Encoder = Encoder(encoder).to(device)
    self.classifier = nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size).to(device)
    self.epochs = config['epochs']
    self.batch_size = config['batch_size']
    self.beta = config['beta']
    self.device = device
    self.optimizer = torch.optim.Adam(
      list(self.classifier.parameters()) + list(self.Encoder.parameters()),
      lr=config['lr'], eps=1e-5)
    self.early_stopper = EarlyStopping(
      patience=self.patience, verbose=False, wandb=self.wandb, name='encoder')
    self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

  def generate_batch(self, episodes):
    total_steps = sum([len(e) for e in episodes])
    print('Total Steps: {}'.format(total_steps))

    sampler = BatchSampler(
      RandomSampler(range(len(episodes)), replacement=True, num_samples=total_steps),
      self.batch_size, drop_last=True
    )
    for indices in sampler:
      episodes_batch = [episodes[x] for x in indices]
      x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
      for episode in episodes_batch:
        t, t_hat = 0, 0
        t, t_hat = np.random.randint(1, len(episode)), np.random.randint(0, len(episode))
        x_t.append(episode[t])
        x_tprev.append(episode[t - 1])
        ts.append([t])
      yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255.

  def do_one_epoch(self, epoch, episodes):
    mode = 'train' if self.encoder.training and self.classifier.training else 'val'
    epoch_loss, steps = 0., 0
    epoch_kl_loss, epoch_nce_loss = 0., 0.
    data_generator = self.generate_batch(episodes)
    for x_t, x_tprev in data_generator:
      z_t, mu, dist = self.Encoder(x_t)
      N = z_t.size(0)
      z_tprev, mu_tprev, dist_tprev = self.Encoder(x_tprev)
      prior_dist = Normal(
        torch.zeros_like(z_t).to(self.device), torch.ones_like(z_t).to(self.device))
      kl_loss = self.beta * kl_divergence(dist, prior_dist).sum(1).mean()

      predictions = self.classifier(z_t)
      logits = torch.matmul(predictions, z_tprev.t())
      nce_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
      loss = kl_loss + nce_loss

      if mode == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      epoch_loss += loss.detach().item()
      epoch_kl_loss += kl_loss.detach().item()
      epoch_nce_loss += nce_loss.detach().item()
      steps += 1

    self.log_results(epoch, epoch_loss / steps, epoch_kl_loss / steps, epoch_nce_loss / steps, prefix=mode)
    if mode == 'val':
      self.early_stopper(-epoch_loss / steps, self.encoder)

  def train(self, tr_eps, val_eps):
    for e in range(self.epochs):
      self.Encoder.train(), self.classifier.train()
      self.do_one_epoch(e, tr_eps)

      self.Encoder.eval(), self.classifier.eval()
      self.do_one_epoch(e, val_eps)

      if self.early_stopper.early_stop:
        break

    torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

  def log_results(self, epoch_idx, epoch_loss, epoch_kl_loss, epoch_nce_loss, prefix=''):
    print(f'{prefix.capitalize()} Epoch: {epoch_idx}, Epoch Loss: {epoch_loss}, {prefix.capitalize()}')

    self.wandb.log({
      prefix + '_loss': epoch_loss,
      prefix + '_kl_loss': epoch_kl_loss,
      prefix + '_nce_loss': epoch_nce_loss,
    }, step=epoch_idx, commit=False)
