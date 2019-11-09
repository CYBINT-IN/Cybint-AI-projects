# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 01:04:51 2019

@author: tanma
"""

from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.densenet import DenseNet121
from keras.models import load_model
from os.path import isfile
from utils import train_val_generator
import matplotlib.pyplot as plt


num_classes = 4
model_name = "vanilla"
train_path = './train'
test_path ='./test'

model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling = 'max')

model.trainable = False

set_trainable = False
for layer in model.layers:
  if layer.name.startswith("conv5" or "conv4"):
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False

m_flatten = model.output
m_dense = Dense(1024,activation='relu')(m_flatten)
m_drop = Dropout(0.5)(m_dense)
m_dense = Dense(1024,activation='relu')(m_drop)
pred_out = Dense(num_classes,activation='softmax')(m_dense)

final_model = Model(model.input,pred_out)
final_model.summary()

tr_model = final_model

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