# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 01:05:28 2020

@author: Tanmay Thakur
"""

import cv2

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from autocorrect import Speller


spell = Speller(lang = "en")

class FilePaths:
	"filenames and paths to data"
	fnCharList = 'model/charList.txt'
	fnAccuracy = 'model/accuracy.txt'
	fnTrain = 'data/'
	fnInfer = 'data/test.png'
	fnCorpus = 'data/corpus.txt'

def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + spell(recognized[0]) + '"')
	print('Probability:', probability[0])

decoderType = DecoderType.BestPath
loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
model = Model(loader.charList, decoderType)    
infer(model, "images_1.jfif")