from keras.applications import VGG16,VGG19
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.models import Model
from resnet_152_keras import resnet152_model
from vgg11_simple import no_transfer_model
import os

def model(model_name, num_classes, is_transfer, num_freeze_layer, weights_path,input_shape):

    # vgg16_model
    conv_base_16 = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
    # conv_base_16.summary()


    # vgg_19 model
    conv_base_19 = VGG19(weights='imagenet',include_top=False,input_shape=input_shape)
    # conv_base_19.summary()

    if not is_transfer or model_name == 'simple':
        model = no_transfer_model(num_classes,input_shape)
        return model
    
    if model_name == 'vgg_16':
        model = conv_base_16
    elif model_name == 'vgg_19':
        model = conv_base_19
    elif model_name == 'resnet_152':
        if not os.path.isfile(weights_path):
            print("Cannot find network weights file")
        model = resnet152_model(weights_path)

    # freeze the given number of layers
    for layer in model.layers[:num_freeze_layer]:
        layer.trainable = False

    # Adding top layers
    m_out = model.output
    m_flatten = Flatten()(m_out)
    m_dense = Dense(1024,activation='relu')(m_flatten)
    m_drop = Dropout(0.5)(m_dense)
    m_dense = Dense(1024,activation='relu')(m_drop)
    pred_out = Dense(num_classes,activation='softmax')(m_dense)

    final_model = Model(model.input,pred_out)
    final_model.summary()

    return final_model