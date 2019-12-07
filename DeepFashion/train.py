# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:19:59 2019

@author: tanma
"""

import shutil
import os
import re
import cv2
import numpy as np
from six.moves import range
import utils as p
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K


splitter = re.compile("\s+")
base_path = './img/'


with open('list_eval_partition.txt', 'r') as eval_partition_file:
    list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
    list_eval_partition = [splitter.split(line) for line in list_eval_partition]
    list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]

# Put each image into the relevant folder in train/test/validation folder
for element in list_all:
    if not os.path.exists(os.path.join(base_path, element[2])):
        os.mkdir(os.path.join(base_path, element[2]))
    if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
        os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
    if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                          element[0].split('/')[0])):
        os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                 element[0].split('/')[0]))
    shutil.move(os.path.join(base_path, element[0]),
                os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0]))

model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

for layer in model_resnet.layers[:-12]:
    layer.trainable = False

x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

x_bbox = model_resnet.output
x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

final_model = Model(inputs=model_resnet.input,
                    outputs=[y, bbox])

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)
final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})

train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y
        
dict_train, dict_val, dict_test = p.get_dict_bboxes()

train_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))

test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200))

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)

tensorboard = TensorBoard(log_dir='./logs')

early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)

checkpoint = ModelCheckpoint('./models/model.h5')


final_model.fit_generator(train_iterator.next(),
                          steps_per_epoch=2000,
                          epochs=200, validation_data=test_iterator.next(),
                          validation_steps=200,
                          verbose=2,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12)

test_datagen = ImageDataGenerator()
test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/test", test_datagen, bounding_boxes=dict_test, target_size=(200, 200))

scores = final_model.evaluate_generator(test_iterator.next(), steps=2000)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))