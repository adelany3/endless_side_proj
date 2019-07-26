#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:19:42 2019

@author: alec.delany
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from model import Autoencoder, Generator, Critic
from dataprep import load

seed = 0
epochs = 15
batch_size = 32
lr = 1e-4
dropout = 0.5
seq_len = 300
gp_lambda = 10
n_critic = 5
n_layers = 20
latent_dim = 100
block_dim = 100
embedding_dim = 200
enc_hidden_dim = 100
dec_hidden_dim = 600
interval = 10
cuda = torch.cuda.is_available()

torch.manual_seed(seed)


def compute_grad_penalty(critic, real, fake):
    B = real.size[0]
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    
    if cuda:
        alpha = alpha.cuda()
    
    sample = alpha*real + (1 - alpha)*fake
    sample.requires_grad_(True)
    score = critic(sample)
    
    outputs = torch.FloatTensor(B, latent_dim).fill_(1.0)
    outputs.requires_grad_(True)
    if cuda:
        outputs = outputs.cuda()
        
    grads = autograd.grad(
    outputs=score,
    inputs=sample,
    grad_outputs=outputs,
    create_graph=True,
    retain_graph=True,
    only_inputs=True
    )[0]
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty

def train(epoch):
    autoencoder.eval()
    generator.train()
    critic.train()
    c_train_loss = 0.
    g_train_loss = 0.
    g_batches = 0
    for i, x in enumerate(train_loader):
        if cuda:
            x = x.cuda()
        
        B = x.size(0)
        c_optimizer.zero_grad()
        noise = torch.from_numpy(np.random.normal(0, 1, (B, latent_dim))).float()
        
        if cuda:
            noise = noise.cuda()
        
        with torch.no_grad(0):
            z_real = autoencoder(x)[0]
        
        z_fake = generator(noise)
        real_score = critic(z_real)
        fake_score = critic(z_fake)
        grad_penalty = compute_grad_penalty(critic, z_real, z_fake)
        c_loss = -torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty
        c_train_loss += c_loss.item()
        c_loss.backward()
        c_optimizer.step()
        
        if i % n_critic == 0:
            g_batches += 1
            g_optimizer.zero_grad()
            fake_score = critic(generator(noise))
            g_loss = -torch.mean(fake_score)
            g_train_loss += g_loss.item()
            
            if interval > 0 and i % interval == 0:
                print('Epoch: {} | Batch: {}/{} ({:.0f}%) | G Loss: {:.6f} | C Loss: {:.6f}'.format(
                epoch, batch_size*i, len(train_loader.dataset),
                100.*(batch_size*i)/len(train_loader.dataset),
                g_loss.item(), c_loss.item()))
                
    g_train_loss /= g_batches
    c_train_loss /= len(train_loader)
    print('* (Train) Epoch: {} | G Loss: {:.4f} | C Loss: {:.4f}'.format(
    epoch, g_train_loss, c_train_loss))
    return (g_train_loss, c_train_loss)

train_loader, vocab = load(batch_size, seq_len)
autoencoder = Autoencoder(enc_hidden_dim, dec_hidden_dim, embedding_dim,
                              latent_dim, vocab.size(), dropout, seq_len)
autoencoder.load_state_dict(torch.load('autoencoder.th', map_location=lambda x,y: x))
generator = Generator(n_layers, block_dim)
critic = Critic(n_layers, block_dim)

g_optimizer = optim.Adam(generator.parameters(), lr = lr)
c_optimizer = optim.Adam(critic.parameters(), lr = lr)
if cuda:
    autoencoder = autoencoder.cuda()
    generator = generator.cuda()
    critic = critic.cuda()
    
best_loss = np.inf

for epoch in range(1, epochs+1):
    g_loss, c_loss = train(epoch)
    loss = g_loss + c_loss
    if loss < best_loss:
        best_loss = loss
        print('Saved')
        torch.save(generator.state_dict(), 'generator.th')
        torch.save(critic.state_dict(), 'critic.th')
    
    
    
    
    
    