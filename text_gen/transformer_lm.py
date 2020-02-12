#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:57:59 2019test.size

@author: alec.delany
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
#from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel
from utils import LRUCache, random_sample

class LanguageModel:
    def predict(self, previous, next_word):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class TextGen(LanguageModel):
    def __init__(self, cache_size = 0):
        self._cache = LRUCache(cache_size, default_value = (None, None))
        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #self.model = GPT2LMHeadModel.from_pretrained('https://storage.googleapis.com/allennlp/models/gpt2-345M-dump')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt-2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt-2')
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
        
        #with torch.no_grad():
        logits, present = self.model(inputs, past = past)
        logits = logits[0, -1]
        
        key = previous if next_word is None else previous + next_word
        self._cache[key] = logits, present
        
        return logits
    
    def __getitem__(self, index):
        return self.tokenizer.decode([index])
    
    def generate(self, seed, max_len, temp):
        
        output = seed
        logits = self.predict(seed)
        
            
        for _ in range(max_len):
            next_id = random_sample(logits, temp)
            #next_id = torch.argmax(torch.nn.functional.softmax(logits/temp, dim = 0)).item()
            next_word_ = self[next_id]
            
            print(next_word_)
            
            if next_word_ == '<|endoftext|>':
                break
            
            logits = self.predict(output, next_word_)
            output += next_word_
        return output
    
    
model = TextGen()
#test = model.predict(previous = 'Franz Kafka awoke one morning to learn that ', next_word = None)  
print(model.generate(seed = 'Franz Kafka awoke one morning to learn that ', max_len = 50, temp = 1.))      