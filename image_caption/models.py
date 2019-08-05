#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:14:09 2019

@author: alec.delany
"""

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    def __init__(self, encoded_img_size = 14):
        super (Encoder, self).__init__()
        self.encoded_img_size = encoded_img_size
        resnet = torchvision.models.resnet101(pretrained = True)
        
        #remove classification layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.adaptive_pool= nn.AdaptiveAvgPool2d((self.encoded_img_size, self.encoded_img_size))
        self.fine_tune()
        
    def forward(self, images):
        '''
        Forward Pass
        :param images: tenor representing image (batch_size, 3, img, img)
        :return: encoded image
        '''
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out
    
    def fine_tune(self, fine_tune = True):
        '''
        Fine tune Conv blocks 2, 3, and 4 of resent
        '''
        
        for p in self.resnet.parameters():
            p.requires_grad = False
        for child in list(self.resnet.children())[5:]:
            for p in child.parameters():
                p.requires_grad = fine_tune
    

            
class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        '''
        :param encoder_dim: size of encoded image
        :param decoder_dim: size of RNN decoder
        :param attenion_dim: size of attention network
        '''
        super(Attention, self).__init__()
        self.encoder_atten = nn.Linear(encoder_dim, attention_dim)
        self.decoder_atten = nn.Linear(decoder_dim, attention_dim)
        self.full_atten = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, encoder_out, decoder_hidden):
        '''
        Forward Pass
        :param encoder_out: tensor of encoded images (batch, n_pixels, encoder_dim)
        :param decoder_hidden: tensor of previous decoder output, hidden state (batch, decoder_dim)
        :return: weighted encoding (from attention), weights
        '''
        att1 = self.encoder_atten(encoder_out) #(batch, n_pixels, atten_dim)
        att2 = self.decoder_atten(decoder_hidden) #(batch, atten_dim)
        att = self.full_atten(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) #(batch, n_pixels)
        alpha = self.softmax(att) #(batch, n_pixels)
        atten_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1) #(batch, encoder_dim)
        
        return atten_weighted_encoding, alpha
    
class DecoderWithAttention(nn.Module):
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim = 2048, dropout = 0.5):
       '''
       :param attention_dim: size of attnetion network
       :param embed_dim: size of embedding
       :param decoder_dim: size of RNN decoder
       :param vocab_size: number of words
       :param encoder_dim: size of encoded images
       :param dropout: dropout %
       '''
       super(DecoderWithAttention, self).__init__()
       
       self.encoder_dim = encoder_dim
       self.embed_dim = embed_dim
       self.decoder_dim = decoder_dim
       self.vocab_size = vocab_size
       self.dropout = dropout
       
       self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
       
       self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
       self.dropout = nn.Dropout(p = self.dropout)
       self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias = True) #LSTM decoder
       self.init_h = nn.Linear(encoder_dim, decoder_dim) #inital hidden state of LSTM
       self.init_c = nn.Linear(encoder_dim, decoder_dim) #initial cell state of LSTM
       self.f_beta = nn.Linear(decoder_dim, encoder_dim) #create a sigmoid-activated gate
       self.sigmoid = nn.Sigmoid()
       self.fc = nn.Linear(decoder_dim, vocab_size) # identify attention scores of vocab
       self.init_weights() #initalize layers with uniform distribution
       
    def init_weights(self):
        '''
        initialize some layers with unifrom stirbution for faster covergence
        '''
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def fine_tune_embeddings(self, fine_tune = True):
        '''
        Allow fine tuning of embedding layer (for pretrained embeddings)
        '''
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
            
    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
        
    def init_hidden_state(self, encoder_out):
        '''
        Initalize hidden state and cell state for LSTM decoder
        
        :param encoder_out: tensor of encoded images (batch, n_pixels, encoder_dim)
        :returns: hidden state, cell state
        '''
        mean_encoder_out = encoder_out.mean(dim = 1) #mean across all pixels
        h = self.init_h(mean_encoder_out) #(batch, decoder_dim)
        c = self.init_c(mean_encoder_out) #(batch, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        '''
        Forward pass
        
        :param encoder_out: tensor of encoded images (batch, enc_img_size, enc_img_size, encoder_dim)
        :param encoded_captions: tensor of encoded captions (batch, max_caption_length)
        :param caption_lengths: tensor of size (batch, 1)
        :return: scores for vocab, sorted captions (encoded), decode lengths, weights, sort indicies
        '''
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) #(batch_size, n_pixels, encoder_dim)
        n_pixels = encoder_out.size(1)
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim = 0, descending = True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        embeddings = self.embedding(encoded_captions)
        
        h, c = self.init_hidden_state(encoder_out) #(batch, decoder_dim)
        
        decode_lengths = (caption_lengths - 1).tolist() #No need to decode <end>
        
        #initialize scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), n_pixels).to(device)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            atten_weight_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) #gating scalar (batch_size_t, encoder_dim)
            atten_weight_encoding = gate * atten_weight_encoding
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], atten_weight_encoding], dim = 1),
                                    (h[:batch_size_t], c[:batch_size_t])) #(batch_t, decoder_dim)
            
            preds = self.fc(self.dropout(h)) #(batch_t, vocab)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind