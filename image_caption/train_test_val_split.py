#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:16:21 2019

@author: alec.delany
"""
import pandas as pd
import numpy as np

file_name = '/Users/alec.delany/Documents/For_Fun/endless_side_proj/image_caption/input_data/flickr30k_images/results.csv'

df = pd.read_csv(file_name, sep = '|')

names = df['image_name'].unique()

test_imgs = np.random.choice(names, 3178, replace = False) #randomly pull 10% of all unique image names
val_imgs = np.random.choice(names, 3178, replace = False) #randomly pull 10% of all unique image names

df['split'] = df['image_name'].apply(lambda x: 'test' if x in test_imgs else 'val' if x in val_imgs else 'train')

test_df = df[df['split'] == 'test']
val_df = df[df['split'] == 'val']
train_df = df[df['split'] == 'train']

test_df.drop(['split'], axis = 1, inplace = True)
val_df.drop(['split'], axis = 1, inplace = True)
train_df.drop(['split'], axis = 1, inplace = True)

for column in test_df.columns:
    test_df = test_df.rename(columns = {column : column.strip()})

for column in train_df.columns:
    train_df = train_df.rename(columns = {column : column.strip()})
    
for column in val_df.columns:
    val_df = val_df.rename(columns = {column : column.strip()})

test_df.to_csv('input_data/flickr30k_images/test.csv', index = False)
train_df.to_csv('input_data/flickr30k_images/train.csv', index = False)
val_df.to_csv('input_data/flickr30k_images/val.csv', index = False)
