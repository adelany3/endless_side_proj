#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:57:59 2019

@author: alec.delany
"""

import torch
from transformers import *
from tqdm import tqdm
from utils import LRUCache, random_sample

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

class LanguageModel():
    def __init__(self, cache_size = 0):
        self._cache = LRUCache(cache_size, default_value = (None, None))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.END_OF_TEXT = self.tokenizer.encoder['<|endoftext|>']
        
    def predict(self, previous, next_word = None):
        
        past_logits, past = self._cache[previous]
        
        if next_word is None and past is not None:
            return past_logits
        elif past is not None:
            token_ids = self.tokenizer.encode(next_word)
        elif next_word is None:
            token_ids = self.tokenizer.encode(previous)
        else:
            token_ids = self.tokenizer.encode(previous) + self.tokenizer.encode(next_word)
            
        inputs = torch.LongTensor([token_ids])
        
        with torch.no_grad():
            logits, present = self.model(inputs, past = past)
        logits = logits[0, -1]
        
        key = [previous if next_word is None else previous + next_word]
        self._cache[key] = logits, present
        
        return logits
    
    def __getitem__(self, index):
        return self.tokenizer.decode([index])
    
    def generate(self, seed, max_len, temp):
        
        output = seed
        logits = self.predict(seed)
        
            
        for _ in tqdm(range(max_len)):
            next_id = random_sample(logits, temp)
            next_word_ = self[next_id]
            
            #print(next_word_)
            
            if next_word_ == '<|endoftext|>':
                break
            
            logits = self.predict(output, next_word_)
            output += next_word_
        return output
    
    
model = LanguageModel()
#test = model.predict(previous = 'What about the', next_word = None)  
print(model.generate(seed = 'To make a great dream come true, the first requirement is a great capacity to dream; the second is', max_len = 50, temp = 1.0))      