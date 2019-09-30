#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:11:16 2019

@author: alec.delany
"""


import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

path = 'input_data/flickr30k_images'
base_filename = 'flickr30k_4_cap_per_img_5_min_word_freq'
checkpoint = '/weights/BEST_checkpoint' + base_filename + '.pth.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

#load model weights
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

with open(os.path.join(path, 'WORDMAP_' + base_filename + '.json'), 'r') as j:
    words2idx = json.load(j)
idx2word = {v:k for k, v in words2idx.items()}
vocab_size = len(words2idx)

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

def evaluate(beam_size):
    '''
    :param beam_size:  Beam size for capiton gen
    :return: BLEU-4 score
    '''
    #very important that batch_size is always 1
    loader = torch.utils.data.DataLoader(CaptionDataset(path, base_filename, 'TEST', 
                                                        transform = transforms.Compose([normalize])), batch_size = 1,
                                                        shuffle = True, num_workers = 1, pin_memory = True)
    reference = []
    hypotheses = []
    
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc = 'Eval @ beam size ' + str(beam_size))):
        k = beam_size
        image = image.to(device)
        
        encoder_out = encoder(image) #(1, enc_img_size, enc_img_size, encoder_dim)
        enc_img_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        
        encoder_out = encoder_out.view(1, -1, encoder_dim) #(1, n_pixels, encoder_dim)
        n_pixels = encoder_out.size(1)
        
        encoder_out = encoder_out.expand(k, n_pixels, encoder_dim) #(beam size, n_pixel, encoder_dim)
        
        k_prev_words = torch.LongTensor(([words2idx['<start>']] * k)).unsqueeze(1).to(device) #(k, 1)
        
        seq = k_prev_words
        
        top_k_scores = torch.zeros(k, 1).to(device)
        
        complete_seqs = []
        complete_seqs_scores = []
        
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)
        
        #s and k move inversely (KE and PE)
        while True:
            
            embeddings = decoder.embedding(k_prev_words).squeeze(1) #(s, embed_dim)
            
            awe, _ = decoder.attention(encoder_out, h) #(s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h)) #(s, encoder_dim)
            awe = gate * awe
            
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim = 1), (h, c)) #(s, decoder_dim)
            
            scores = decoder.fc(h) #(s, vocab_size)
            scores = F.log_softmax(scores, dim = 1)
            
            scores = top_k_scores.expands_as(scores) + scores #(s, vocab_size)
            
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) #(s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) #(s)
                
            prev_word_idx = top_k_words / vocab_size
            next_word_idx = top_k_words % vocab_size
            
            #Add word to seq
            seq = torch.cat([seq[prev_word_idx], next_word_idx.unsqueeze(1)], dim = 1) #(s, step + 1)
            
            #drop seqs that didn't terminate with <end>
            faulty_idx = [idx for idx, next_word in enumerate(seq.unsqueeze(0)) if next_word != words2idx['<end>']]
            complete_idx = list(set(range(len(seq))) - set(faulty_idx))
            
            if len(complete_idx) > 0:
                complete_seqs.extend(seq[complete_idx].to_list())
                complete_seqs_scores.extend(top_k_scores[complete_idx])
            k -= len(complete_idx)
            
            if k == 0:
                break
            seq = seq[faulty_idx]
            h = h[prev_word_idx[faulty_idx]]
            c = c[prev_word_idx[faulty_idx]]
            encoder_out = encoder_out[prev_word_idx[faulty_idx]]
            top_k_scores = top_k_scores[faulty_idx].unsqueeze(1)
            k_prev_words = next_word_idx[faulty_idx].unsqueeze(1)
            
            if step > 40:
                break #Something has gone quite wrong
            step += 1
            
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        
        img_caps = allcaps[0].tolist()
        img_captions = list(map(lambda c: [w for w in seq if w not in {words2idx['<start>'], words2idx['<end>'], words2idx['<pad>']}]))
        reference.append(img_captions)
        
        hypotheses.append([w for w in seq if w not in {words2idx['<start>'], words2idx['<end>'], words2idx['<pad>']}])
        
        assert len(reference) == len(hypotheses)
        
        bleu4 = corpus_bleu(reference, hypotheses)
        
        return bleu4
    
for beam_size in [1, 2, 3, 4, 5, 6]:

    print('\nBLEU-4 score @ beam size %d if %.4f' % (beam_size, evaluate(beam_size)))
            
            
        
        
        
