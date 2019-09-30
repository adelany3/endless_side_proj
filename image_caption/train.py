#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:16:28 2019

@author: alec.delany
"""

    
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from dataset import CaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import os
import json



path = 'input_data/flickr30k_images'
base_filename = 'flickr30k_4_cap_per_img_5_min_word_freq'

emb_dim = 512
atten_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 32
workers = 1
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.
alpha_c = 1. #reg parameter for "doubly stochastic attention"
best_bleu4 = 0.
print_freq = 100
fine_tune_encoder = True
checkpoint = None


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    '''
    Trains for one epoch
    '''
    print('====* Training for Epoch %i *====' % epoch)
    
    decoder.train()
    encoder.train()
    
    losses = AverageMeter()
        
    for i, (img, caps, caplens) in enumerate(tqdm(train_loader)):
        
        img = img.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        img = encoder(img)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(img, caps, caplens)
        
        targets = caps_sorted[:, 1:]
        
        scores= pack_padded_sequence(scores, decode_lengths, batch_first = True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)

        loss = criterion(scores.data, targets.data)
        
        #Doubly stochastic attention reg
        loss += alpha_c * ((1. - alphas.sum(dim = 1)) ** 2).mean()   #MSE
        
        #back Prop
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)
        
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
            
        losses.update(loss.item(), sum(decode_lengths))
        
    print('Epoch: ', epoch)
    print('Loss_val: ', losses.val)
    print('Loss_avg: ', losses.avg)
    
def validate(val_loader, encoder, decoder, criterion):
    '''
    Validates one epoch
    '''
    
    decoder.eval()
    if encoder is not None:
        encoder.eval()
        
    losses = AverageMeter()
    
    actuals = [] #true captions
    yhat = [] #predicted captions
    
    with torch.no_grad():
        for i, (img, caps, caplens, allcaps) in enumerate(tqdm(val_loader)):
            
            img = img.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            
            if encoder is not None:
                img = encoder(img)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(img, caps, caplens)
            
            targets = caps_sorted[:, 1:]
            
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)
            
            scores = scores.data
            targets = targets.data
            
            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            losses.update(loss.item(), sum(decode_lengths))
        
            #print('Loss_val: ', losses.val)
            #print('Loss_avg: ', losses.avg)
            
            #Collecrt actuals
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(map(lambda x: [w for w in x if w not in {words2idx['<start>'], words2idx['<pad>']}], 
                                        img_caps))
                actuals.append(img_captions)
                
            #Collect yhat
            _, preds = torch.max(scores_copy, dim = 2)
            #preds = preds.to_list()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            yhat.extend(preds)
            
            assert len(actuals) == len(yhat)
            
        bleu4 = corpus_bleu(actuals, yhat)
        
        print('Loss_val: ', losses.val)
        print('Loss_avg: ', losses.avg) 
        
    return bleu4
 

with open(os.path.join(path, 'WORDMAP_' + base_filename + '.json'), 'r') as j:
    words2idx = json.load(j)

if checkpoint is None:
    decoder = DecoderWithAttention(attention_dim = atten_dim, embed_dim = emb_dim, decoder_dim = decoder_dim,
                                   vocab_size = len(words2idx), dropout = dropout)
    decoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()), lr = decoder_lr)
    
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    if fine_tune_encoder:
        encoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), 
                                             lr = encoder_lr)
    else:
        encoder_optimizer = None
        
else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    
    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), lr = encoder_lr)

decoder = decoder.to(device)
encoder = encoder.to(device)

criterion = nn.CrossEntropyLoss().to(device)

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.255])

train_loader = torch.utils.data.DataLoader(CaptionDataset(path, base_filename, 'TRAIN', transform = transforms.Compose([normalize])),
                                          batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True)
val_loader = torch.utils.data.DataLoader(CaptionDataset(path, base_filename, 'VAL', transform = transforms.Compose([normalize])),
                                          batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True)

for epoch in range(start_epoch, epochs):
    #early stopping if no imporvement after 10 epochs
    if epochs_since_improvement == 10:
        break
    if (epochs_since_improvement > 0) and (epochs_since_improvement % 4 == 0):
        adjust_learning_rate(decoder_optimizer, 0.8)
        if fine_tune_encoder:
            adjust_learning_rate(encoder_optimizer, 0.8)
            
    train(train_loader = train_loader, encoder = encoder, decoder = decoder, criterion = criterion, 
                          encoder_optimizer = encoder_optimizer, decoder_optimizer = decoder_optimizer, epoch = epoch)
    
    recent_bleu4 = validate(val_loader = val_loader, encoder = encoder, decoder = decoder, criterion = criterion)
    
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print ('Epochs since imrpovements: ', epochs_since_improvement)
    else:
        epochs_since_improvement = 0
    
    save_checkpoint(base_filename, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4, is_best)