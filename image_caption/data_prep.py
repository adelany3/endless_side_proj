#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:34:14 2019

@author: alec.delany
"""

import numpy as np
import string
from collections import Counter
from utils import *
import os
import h5py
import json
from scipy.misc import imread, imresize
from tqdm import tqdm
from random import seed, choice, sample

path = 'input_data/flickr30k_images'
files = ['train.csv', 'test.csv', 'val.csv']

train_images = []
train_image_captions = []
val_images = []
val_image_captions = []
test_images = []
test_image_captions = []   
master = {}
word_freq = Counter()
min_word_freq = 5
#max_len = 100
captions_per_image = 4

for file in files:
    name = file[:-4]
    master[name] = read_and_clean_data(os.path.join(path, file))
print('Generated Master Dictionary...')    
for dataset in master.keys():
    for img, captions in master[dataset].items():
        if dataset == 'train':
            train_images.append(img)
            train_image_captions.append(captions)
        elif dataset == 'test':
            test_images.append(img)
            test_image_captions.append(captions)
        elif dataset == 'val':
            val_images.append(img)
            val_image_captions.append(captions)
        for caption in captions:
            word_freq.update(caption.split(' '))
            
assert len(train_images) == len(train_image_captions)
assert len(val_images) == len(val_image_captions)
assert len(test_images) == len(test_image_captions)

words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
words2idx = {k: v + 1 for v, k in enumerate(words)}
words2idx['<unk>'] = len(words2idx) + 1
words2idx['<start>'] = len(words2idx) + 1
words2idx['<end>'] = len(words2idx) + 1
words2idx['<pad>'] = 0

print('Total words: ', len(words2idx))
base_filename = 'flickr30k_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

with open(os.path.join(path, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
    json.dump(words2idx, j)

print('Completed Word Map')

max_len = 200
seed(111)
for impaths, imcaps, split in [(train_images, train_image_captions, 'TRAIN'),
                               (val_images, val_image_captions, 'VAL'),
                               (test_images, test_image_captions, 'TEST')]:
    
    with h5py.File(os.path.join(path, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
        h.attrs['captions_per_image'] = captions_per_image
        
        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
        
        print("\nReading %s images and captions, storing to file...\n" % split)
        
        enc_captions = []
        caplens = []
        
        for i, name in enumerate(impaths):
            
            #imcaps[i] = [x for x in imcaps[i] if len(x) <= max_len]
            
            #if len(imcaps[i]) >= 2:
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k = captions_per_image)
                
            assert len(captions) == captions_per_image
            
            img = imread(os.path.join(path + '/flickr30k_images', impaths[i]))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis = 2)
            img = imresize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255
            
            images[i] = img
            
            for j, c in enumerate(captions):
                enc_c = [words2idx['<start>']] + \
                        [words2idx[word] if word in words2idx.keys() else words2idx['<unk>'] for word in c.split(' ') ] + \
                        [words2idx['<end>']] + [words2idx['<pad>']] * (max_len - len(c.split(' ')))
                
                enc_captions.append(enc_c)
                caplens.append((len(c.split(' ')) + 2)) #Adding <start> and <end>
                assert len(enc_c) == (max_len + 2)
        assert max(caplens) <= (max_len + 2)
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
        
        with open(os.path.join(path, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)
        with open(os.path.join(path, split + '_CAPTLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
        
        