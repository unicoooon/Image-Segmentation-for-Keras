import os
import sys
import math
import random as rn

import cv2
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.optimizers import Adadelta, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from model  import fcnZoo



loss_name = "categorical_crossentropy"

def get_model(input_shape=(256,256,7),num_classes=2):
    
    inputs = Input(shape=input_shape)
    
    #base = fcnZoo.get_fcn_vgg16_32s(inputs, num_classes)
    base = fcnZoo.get_fcn_vgg16_16s(inputs, num_classes)
    #base = models.get_fcn_vgg16_8s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_unet(inputs, NUMBER_OF_CLASSES)
    #base = fcnZoo.get_segnet_vgg16(inputs, num_classes) 
    model = Model(inputs=inputs, outputs=base)
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
    
    #print(model.summary())
    #sys.exit()
    
    return model
