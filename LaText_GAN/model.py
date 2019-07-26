#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:04:42 2019

LSTM based GAN for text generation
based on https://arxiv.org/pdf/1810.06640.pdf

I anticipate this to be very unstable.
DS version of "hold my beer, watch this"

@author: alec.delany
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self, enc_hidden_dim, dec_hidden_dim, embedding_dim, latent_dim, vocab_size,
                 dropout, seq_len):
        super().__init__()
        
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, enc_hidden_dim)
        self.fc1 = nn.Linear(enc_hidden_dim, latent_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(latent_dim, dec_hidden_dim)
        self.decoder = nn.LSTM(dec_hidden_dim, dec_hidden_dim)
        self.fc3 = nn.Linear(dec_hidden_dim, vocab_size)
        
    def encode(self, x):
        x = self.embedding(x).permute(1, 0, 2) #Why?
        _, (hidden, _) = self.encoder(x)
        z = self.fc1(hidden)
        z = self.dropout(x)
        return z
    
    def decode(self, z):
        z = self.fc2(z)
        out, _ = self.decoder(z.repeat(self.seq_len, 1, 1), (z, z))
        out = out.permute(1, 0, 2)
        logits = self.fc3(out)
        return logits.transpose(1, 2)
    
    def forward(self, x):
        z = self.encode(x)
        logits = self.decode(z)
        return (z.squeeze(), logits)
    

class Block(nn.Module):
    
    def __init__(self, block_dim):
        super().__init__()
        
        self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim))
        
    def forward(self, x):
        return self.net(x) + x
    
class Generator(nn.Module):
    
    def __init__(self, n_layers, block_dim):
        super().__init__()
        
        self.net = nn.Sequential( *[Block(block_dim) for _ in range(n_layers)] )
        
    def forward(self, x):
        return self.net(x)
    
class Critic(nn.Module):
    
    def __init__(self, n_layers, block_dim):
        super().__init__()
        
        self.net = nn.Sequential( *[Block(block_dim) for _ in range(n_layers)] )
        
    def forward(self, x):
        return self.net(x)
        