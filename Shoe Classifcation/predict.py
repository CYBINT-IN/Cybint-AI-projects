# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 00:07:11 2019

@author: tanma
"""
from utils import train_val_generator

train_path = './train'
test_path ='./test'

# train and test generator
train_gen, val_gen = train_val_generator(32,train_path,test_path)