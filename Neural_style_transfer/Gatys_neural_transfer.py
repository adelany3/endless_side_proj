#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:43:43 2019

@author: alec.delany
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import scipy.misc
import numpy as np

class Gram(nn.Module):
    def forward(self, input):
        a, b, c, d, = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    
class TVLoss(nn.Module):
    """
    Total variation loss.
    """
    def __init__(self):
        super(TVLoss, self).__init__()
    '''
    def forward(self, yhat, y):
        bsize, chan, height, width = y.size()
        errors = []
        for h in range(height-1):
            dy = torch.abs(y[:,:,h+1,:] - y[:,:,h,:])
            dyhat = torch.abs(yhat[:,:,h+1,:] - yhat[:,:,h,:])
            error = torch.norm(dy - dyhat, 1)
            errors.append(error)

        return sum(errors) / height
    '''
    def forward(self, img, weight):
        w_variance = torch.sum((img[:,:,:,1:] - img[:,:,:,:-1])**2)
        h_variance = torch.sum((img[:,:,1:,:] - img[:,:,:-1,:])**2)
         
        loss = weight * (w_variance + h_variance)
    
        return loss

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
class StyleCNN(object):
    def __init__(self, style, content, mew_img, alpha, beta, reg, norm_mean, norm_std):
        super(StyleCNN, self).__init__()
        
        self.style = style
        self.content = content
        self.new_img = nn.Parameter(new_img.data)
        self.content_layers = ['conv_4', 'conv_5']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = alpha
        self.style_weight = beta
        self.reg_weight = reg
        self.mean = norm_mean
        self.std = norm_std
        self.loss_network = models.vgg19(pretrained = True)
        
        for i, layer in enumerate(self.loss_network.features):
            if isinstance(layer, nn.MaxPool2d):
                self.loss_network.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.gram = Gram()
        self.total_var_loss = TVLoss()
        self.loss = nn.MSELoss()
        #self.optimizer = optim.LBFGS([self.new_img])
        #self.optimizer = optim.Adam([self.new_img])
        self.optimizer = optim.RMSprop([self.new_img])
        self.cuda = torch.cuda.is_available()
        '''
        if self.cuda:
            self.loss_network.cuda()
            self.gram.cuda()
            self.loss.cuda()
            self.optimizer.cuda()
            self.total_var_loss.cuda()
            self.mean.cuda()
            self.std.cuda()
        '''
            
    def train(self):
        def closure():
            self.optimizer.zero_grad()
            
            new_img = self.new_img.clone()
            new_img.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()
            
            #normalization = Normalization(self.mean, self.std)
            content_loss = 0
            style_loss = 0
            tv_loss = 0
            
            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace = False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                '''
                if self.cuda():
                    layer.cuda()
                '''
                
                new_img, content, style = layer.forward(new_img), layer.forward(content), layer.forward(style)
                                
                if isinstance(layer, nn.Conv2d):
                    name = 'conv_' + str(i)
                    
                    tv_loss += self.total_var_loss.forward(new_img, self.reg_weight)
                    
                    if name in self.content_layers:
                        content_loss += self.loss(new_img * self.content_weight, content.detach() * self.content_weight)
                        #tv_loss += self.total_var_loss.forward(new_img, self.reg_weight)
                    if name in self.style_layers:
                        new_img_gram = self.gram.forward(new_img)
                        style_gram = self.gram.forward(style)
                        style_loss += self.loss(new_img_gram * self.style_weight, style_gram.detach() * self.style_weight)
                        #tv_loss += self.total_var_loss.forward(new_img, self.reg_weight)
                if isinstance(layer, nn.ReLU):
                    i += 1
            
            total_loss = content_loss + style_loss# + tv_loss
            total_loss.backward()
            return total_loss
        self.optimizer.step(closure)
        return self.new_img

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((img_size[0], img_size[1]))
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def save_image(inp, path):
    image = inp.data.clone().cpu()
    image = image.view(3, img_size[0], img_size[1])
    image = unloader(image)
    scipy.misc.imsave(path, image)
    #imageio.imwrite(path, image)
   
#aspect_ratio = 1619/1080
#aspect_ratio = 3264/448
#aspect_ratio = 1
#size_seed = 256
#img_size = (size_seed, int(np.round_(size_seed*aspect_ratio, 0)))
vgg19_norm_mean = torch.tensor([0.485, 0.456, 0.406])
vgg19_norm_std = torch.tensor([0.229, 0.224, 0.225])

#loader = transforms.Compose([transforms.Resize((img_size[0], img_size[1])), transforms.ToTensor()])
#unloader = transforms.ToPILImage()

#dtype = torch.FloatTensor

'''
for file in ['monet.jpg', 'leonid_afremov_bride.jpg', 'picasso_mandolin.jpg', 'Pollock_neuron.jpg', 'Matisse.jpg', 'im_blau.jpg',
             'Duchamp_descending.jpg']:
'''
'''
#for beta in [1000, 2500, 5000, 75000, 10000]:
counter = 0
for reg in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    file = 'leonid_afremov_bride.jpg'
    style = image_loader('transfer_data/' + file).type(dtype)
    content = image_loader('transfer_data/IMG_2738.JPG').type(dtype)
    new_img = image_loader('transfer_data/IMG_2738.JPG').type(dtype)
    new_img.data = torch.randn(new_img.data.size()).type(dtype)
    
    n_epochs = 51
    style_cnn = StyleCNN(style, content, new_img, alpha = 1, beta = 1000, reg = reg, norm_mean = vgg19_norm_mean, norm_std = vgg19_norm_std)
    #print(file[:-4])
    for i in range(n_epochs):
        new_img = style_cnn.train()
        print('Epoch number: %d' % (i))
        
        if i % 10 == 0:
            
            #path = 'transfer_data/Gatys_output/450/' + file[:-4] + '_epoch_%d_with_tv_from_org.png' % (i)
            
            path = 'transfer_data/Gatys_output/tv_test/' + str(counter) + '_epoch_%d_with_tv.style_from_org.png' % (i)
            new_img.data.clamp_(0, 1)
            save_image(new_img, path)
    counter += 1
''' 
series = {#'Erin': ['IMG_2738.JPG', 'leonid_afremov_bride.jpg'],
          'Frank': ['Frank.JPG', 'im_blau.jpg'],
          #'Network': ['CNN_arch2.jpg', 'PCB3.jpg'],
          #'Python': ['python-logo2.jpg', 'meta_code2.jpg']
          }

for name in series.keys():
    files = series[name]
    
    aspect_ratio = Image.open('transfer_data/' + files[0]).size[0]/Image.open('transfer_data/' + files[0]).size[1]
    size_seed = 256
    img_size = (size_seed, int(np.round_(size_seed*aspect_ratio, 0)))
    loader = transforms.Compose([transforms.Resize((img_size[0], img_size[1])), transforms.ToTensor()])
    unloader = transforms.ToPILImage()

    dtype = torch.FloatTensor
    
    style = image_loader('transfer_data/' + files[1]).type(dtype)
    content = image_loader('transfer_data/' + files[0]).type(dtype)
    new_img = image_loader('transfer_data/' + files[0]).type(dtype)
    new_img.data = torch.randn(new_img.data.size()).type(dtype)
    
    n_epochs = 1501
    style_cnn = StyleCNN(style, content, new_img, alpha = 1, beta = 1800, reg = 1e2, norm_mean = vgg19_norm_mean, norm_std = vgg19_norm_std)
    #print(file[:-4])
    for i in range(n_epochs):
        new_img = style_cnn.train()
        #print('Epoch number: %d' % (i))
        
        if i % 100 == 0:
            
            print('Epoch number: %d' % (i))
            path = 'transfer_data/Gatys_output/256/RMSprop_beta1800_reg.1e2_epoch_{0}_with_all.tv_{1}.png'.format(i, name)
            new_img.data.clamp_(0, 1)
            save_image(new_img, path)