#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:41:48 2020

@author: alec.delany
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ 
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert (logits.dim() == 1)  # batch size 1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.9, 
                    repetition_penalty=0.9):
    context = torch.tensor(context, dtype=torch.long)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {"input_ids": generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.0)

            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


class LanguageGenerator:
    def __init__(self, num_words, weights):
        self.num_words = num_words
        self.tokenizer = GPT2Tokenizer.from_pretrained(weights)
        model = GPT2LMHeadModel.from_pretrained(weights)
        model.eval()
        self.model = model

    def predict(self, text):
        indexed_tokens = self.tokenizer.encode(text)
        output = sample_sequence(self.model, self.num_words, indexed_tokens, temperature = 0.75, repetition_penalty = 0.8)
        return self.tokenizer.decode(
            output[0, 0:].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
    
if __name__ == '__main__':
    txt = open('the_raven.txt').read()
    output = LanguageGenerator(120, 'distilgpt2').predict(txt[:1000])
    print('\n')
    print(output[950:])