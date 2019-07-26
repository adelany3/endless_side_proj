#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:21:06 2019

@author: alec.delany
"""

import torch
import numpy as np
import collections
import pandas as pd
import numpy as np
import codecs
from torch.utils.data import DataLoader, Dataset

class Vocab:
    
    def __init__(self, corpus):
        self.words = self.build(corpus)
        self.encoding = {w:i for i, w in enumerate(self.words, 1)}
        self.decoding = {i:w for i, w in enumerate(self.words, 1)}
        
    def build(self, corpus, clip = 1):
        vocab = collections.Counter()
        
        for word in corpus:
            vocab.update(word)
        
        for word in vocab.keys():
            if vocab[word] < clip:
                vocab.pop[word]
                
        return list(sorted(vocab.keys()))
    
    def size(self):
        
        assert len(self.encoding) == len(self.decoding)
        return len(self.encoding)
    
class Corpus:
    
    def __init__(self, seq_len = 80):
        self.seq_len = seq_len
        self.abstract, self.titles = self._load()
        self.vocab = Vocab(self.abstract)
        
    def _load(self):
        df = pd.read_json(codecs.open('../data/arxivData.json', 'r', 'utf-8'))
        
        abstracts = [x.split(' ') for x in df['summary'].values]
        titles = df['title'].values
        
        return abstracts, titles
    
    def pad(self, sample):
        l, r = 0, self.seq_len - len(sample)
        return np.pad(sample, (0, r), 'constant')
    
    def __len__(self):
        return len(self.abstract)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.pad(self.abstract[i]))
    
def load(batch_size, seq_len):
    ds = Corpus(seq_len)
    
    return (DataLoader(ds, batch_size, shuffle = True), ds.vocab)
    
    
        
        
        
        
        
        