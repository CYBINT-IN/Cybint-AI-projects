# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 01:05:28 2020

@author: Tanmay Thakur
"""

import cv2
import os

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from autocorrect import Speller
from main import infer, FilePaths


spell = Speller(lang = "en")

decoderType = DecoderType.BestPath
loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
model = Model(loader.charList, decoderType)    

imgFiles = os.listdir('data_/')
for i in imgFiles:
    images = os.listdir('out/%s'%i)
    for j in images:
        if(j != "summary.png"):
            infer(model,"out/"+ i + "/" + j)