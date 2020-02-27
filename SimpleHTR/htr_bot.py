# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:00:12 2019

@author: tanma
"""
import os
import logging
import urllib.request

from segment import segment_main
from telegram import ChatAction
from telegram.ext import Updater, MessageHandler, Filters

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from autocorrect import Speller
from main import infer, FilePaths


spell = Speller(lang = "en")

decoderType = DecoderType.BestPath
loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
model = Model(loader.charList, decoderType) 


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def download_image(url,name):
    fullname = str(name)+".jpeg"
    urllib.request.urlretrieve(url,fullname)


def some_func(bot, update):
    pass
    if not update.effective_message.photo:
        update.effective_message.reply_text(text = "This bot is only capable of Computer Vision Tasks!")
    else:
        msg = update.effective_message
        file_id = msg.photo[-1].file_id
        photo = bot.get_file(file_id)
        download_image(photo["file_path"],'wassup')
        text = infer(model, 'wassup.jpeg')
        update.effective_message.reply_text(text = text)

        
def main():
    updater = Updater('1059913469:AAHPrHeLuqVEz-UbRezjyYJQ_swoHrQ-_QM')
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.all, some_func))
    updater.start_polling()
    updater.idle()
    
if __name__ == '__main__':
    main()