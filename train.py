from __future__ import print_function
from sklearn.model_selection import train_test_split
import os
#import matplotlib.pyplot as plt
import argparse
from model.segnet import SegNet
import model.maskrcnn as modellib
from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck
from model.deeplabv3p import Deeplabv3
from model.deeplabs import deeplabv3_plus
from model.fcnZooModel import get_model
from model.Icnet import build_bn,build
from model.refinenet import refinenet,refineNet
from model.unet import unet
from model.fcn import fcn_8s
from model.lstm import model
from model.pspnet import pspnet50
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
# Current python dir path
from mrcnn.config import Config
from model.Icnet import build_bn,build
from model.fcnZooModel import get_model
import h5py
import numpy as np
dir_path = os.path.dirname(os.path.realpath('__file__'))
import datetime
import random
img_cols=256
img_rows=256
num_channels=7
num_mask_channels=2
batch_size=4
# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'lstm','fcn32','unet', 'pspnet','deeplabv3p','deeplabv3','maskrcnn','segnet','refinenet','icnet'],
                    help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg
config = Config()
labels=2
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 256, 7), num_classes=labels,
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(256,256, 7), num_classes=labels,
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 256, 7), num_classes=labels, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "deeplabv3p":
    model = Deeplabv3(input_shape=(256, 256, 3), classes=labels)
elif model_name == "deeplabv3":
    model = deeplabv3_plus(input_shape=(256,256,7), num_classes=labels)
elif model_name == "maskrcnn":
    modelt= modellib.MaskRCNN(mode = 'training',config=config,model_dir=MODEL_DIR)
elif model_name =='refinenet':
    model = refinenet(input_shape=(256,256,5),num_classes=lebels)
    #model =build_network_resnet101(inputHeight=256,inputWidth=256,n_classes=len(labels))
    #model = build_network_resnet101_stack(inputHeight=256,inputWidth=256,n_classes=len(labels),nStack=2)
elif model_name =="segnet":
    model =SegNet(input_shape=(256,256,5),classes=labels)
elif model_name =="fcn32":
    model = get_model(input_shape=(256,256,7),num_classes=labels)
elif model_name =='icnet':
    model = build_bn(input_shape=(256,256,7),n_classes=labels)
elif model_name =="lstm":
    model = model(shape=(256,256,7),num_classes=labels)
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, img_rows, img_cols,num_channels))
    y_batch = np.zeros((batch_size, img_rows, img_cols,num_mask_channels))
    X_height = X.shape[1]
    X_width = X.shape[2]
    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, random_height: random_height + img_rows, random_width: random_width + img_cols,:]
        X_batch[i] = np.array(X[random_image,random_height: random_height + img_rows, random_width: random_width + img_cols,:])
    return X_batch, y_batch

def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb


        yield X_batch, y_batch#[:, :, 16:16 + img_rows - 32, 16:16 + img_cols - 32]
import cv2
from skimage.external import tifffile
imgpath =["/home/langyan/NewTrain/1814trainImg.png","/home/langyan/NewTrain/1815trainImg.png",
          "/home/langyan/NewTrain/1816trainImg.png","/home/langyan/NewTrain/1817trainImg.png"]
tifpath = ["/data/test_h/AlignData/118030/LC81180302014212LGN00_merge.tif","/data/test_h/AlignData/118030/20150803_118030.tif","/data/test_h/AlignData/118030/20160821_118030.tif","/data/test_h/AlignData/118030/20170824_118030.tif"]
imgSize = cv2.imread(imgpath[0],0)
from keras.utils import np_utils
def getData(maskpath,tifpath):
    tmpData = np.zeros((4,imgSize.shape[0],imgSize.shape[1],7))
    tmpMask = np.zeros((4,imgSize.shape[0],imgSize.shape[1],2))
    for i in range(4):
        tmpData[i,:,:,:]=tifffile.imread(tifpath[i])
        tmpMask[i,:,:,:] = np_utils.to_categorical(cv2.imread(maskpath[i],0),2)
    return tmpData/255,tmpMask
X_train,y_train = getData(imgpath,tifpath)

#f2 = h5py.File(os.path.join(data_path,"dysmall.h5"),'r')
#XX_train = f2['train']
#print(XX_train.shape)
#yy_train = np.array(f2['train_mask'])

#datat = np.zeros((1,8277,7663,3))
#labelt = np.zeros((1,8277,7663,3))

#datat[:,0:8181,0:7362,:]=XX_train
#labelt[:,0:8181,0:7362,:]=yy_train

#tdata = np.concatenate([X_train,datat],0)
#tlabel = np.concatenate([y_train,labelt],0)

'''
print(y_train.shape)
f2 = h5py.File(os.path.join(data_path,"secondh5.h5"),'r')
XX_train = f2['train']
print(XX_train.shape)
yy_train = np.array(f2['train_mask'])
print(yy_train.shape)
f3 = h5py.File(os.path.join(data_path,"thirdh5.h5"),'r')
x3 = f3['train']
y3 = np.array(f3['train_mask'])

data3 = np.zeros((1,2122,2474,5))
label3 = np.zeros((1,2122,2474,16))

data3[:,0:773,0:1584,:]=x3
label3[:,0:773,0:1584,:]=y3

data_one = np.zeros((1,2122,2474,5))
label_one = np.zeros((1,2122,2474,16))
data_one[:,0:1082,0:1476,:] = XX_train
label_one[:,0:1082,0:1476,:] = yy_train
tt_data = np.concatenate([X_train,data_one,data3],0)
tt_label = np.concatenate([y_train,label_one,label3],0)
'''
#datai,labeli = read_data(file_list,mask_list)
#print(np.unique(datai))
#print(datai[0,500:510,500:510,:])
#print(datai[0,500:510,500:510,:])
#tt_data = np.reshape(tt_data,(8,1061,1237,5))
#tt_label =np.reshape(tt_label,(8,1061,1237,16))
#print(tdata.shape)		
#print(tlabel.shape)
#train_x,val_x,train_y,val_y=train_test_split(np.array(X_train),y_train,test_size=0.25) 

#y_train = np.expand_dims(y_train, 1)
#batchx,batchy = form_batch(X_train,y_train,batch_size)
#gendata = ImageDataGenerator(vertical_flip=True,horizontal_flip=True)

# Define callbacks
checkpoint = ModelCheckpoint(filepath='h5File/'+model_name + '_model_weight_fresh_next.h5',
                             save_best_only=True,
                             save_weights_only=True)
history  = model.fit_generator(batch_generator(X_train,y_train,batch_size),steps_per_epoch=400,
                               validation_data = batch_generator(X_train,y_train,batch_size),validation_steps=2,
                               callbacks=[checkpoint],epochs=50,verbose=1)

