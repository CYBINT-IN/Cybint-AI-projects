# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:39:22 2019

@author: tanma
"""

from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras.optimizers import Adam
from models import model
from utils import train_val_generator
from matplotlib import pyplot as plt
from keras.models import load_model
from os.path import isfile


choices=['vgg_16','vgg_19','resnet_152','simple']
model_name = choices[2]

is_transfer = True
num_freeze_layer = 600
num_classes = 4
weights_path = 'resnet152_weights_tf.h5'
input_shape = (224,224,3) # Input Shape for Resnet152

train_path = './train'
test_path ='./test'



# model for traning
tr_model = model(model_name,num_classes,is_transfer,
                num_freeze_layer,weights_path,input_shape)

# train and test generator
train_gen, val_gen = train_val_generator(32,train_path,test_path)

# load last model if exists
model_name = model_name+'.h5'
if isfile(model_name):
    print('Loading previously trained weights and continue traning.....')
    tr_model = load_model(model_name)
else:
    print('No saved weights found.')

# model saving
checkpoint = ModelCheckpoint(model_name+'.h5',monitor='val_acc',verbose=1,save_best_only=True)
early_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=10,verbose=1,mode='auto')

# Compile the model
tr_model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-5),metrics=['accuracy'])

# train the model
history = tr_model.fit_generator(
                train_gen,
                steps_per_epoch=1400,
                epochs=30,
                validation_data = val_gen,
                validation_steps = 250,
                callbacks = [checkpoint,early_stop])

# plot the results
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.savefig(model_name+'.jpg')

tr_model.save('vanilla.h5')