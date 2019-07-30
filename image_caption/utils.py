#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:53:29 2019

@author: alec.delany
"""

import pandas as pd
import numpy as np
import string

def read_and_clean_data(file):
    df = pd.read_csv(file)
    
    out_dict = {}
    table = str.maketrans('','',string.punctuation)
    for image in df['image_name']:
        temp = df[df['image_name'] == image]
        out_dict[image] = []
        for caption in temp['comment'].values:
            caption = caption.split(' ')
            caption = [x.lower() for x in caption]
            caption = [x.translate(table) for x in caption]
            caption = [x for x in caption if len(x)>1]
            caption = [x for x in caption if x.isalpha()]
            caption = ' '.join(caption)
            out_dict[image].append(caption)
    return out_dict

test = read_and_clean_data('input_data/flickr30k_images/val.csv')