# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:25:50 2019

@author: tanma
"""

from fastai import *
from fastai.vision import *

torch.cuda.set_device(0)

path = Path('data/cloth_categories')

classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']

single_img_data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(),
                                                     size=150).normalize(imagenet_stats)

learn = create_cnn(single_img_data,models.resnet34)

learn.load('stage-1_sz-150')

"""
# if you have export.pkl file, do this straight away
learn = load_learner('data/cloth_categories/stage-1_sz-150')
"""

IMG_FILE_SRC =  path/"test_images/test-1.jpeg"

show_image(open_image(IMG_FILE_SRC))

_,_,losses = learn.predict(open_image(IMG_FILE_SRC))

predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)

print (predictions[:5])