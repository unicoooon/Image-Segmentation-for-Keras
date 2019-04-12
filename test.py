from __future__ import print_function

import argparse
import cv2
import numpy as np
import tifffile
from model.refinenet import refinenet,refineNet
from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.deeplabv3p import Deeplabv3
from model.deeplabs import deeplabv3_plus
from model.segnet import SegNet
import os
import numpy as np
import cv2
import tifffile
os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet','refinenet','deeplabv3p','deeplabv3','segnet'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path

labels = 2
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 256, 5), num_classes=labels, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet":
    model = unet(input_shape=(256, 256, 7), num_classes=labels, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 256, 5), num_classes=labels, lr_init=1e-3, lr_decay=5e-4)
elif model_name =='deeplabv3p':
    model = Deeplabv3(input_shape=(256, 256, 5), classes=labels)
elif model_name == "deeplabv3":
    model = deeplabv3_plus(input_shape=(256, 256, 5), num_classes=labels)
elif model_name =="segnet":
    model =SegNet(input_shape=(256,256,5),classes=labels)
elif model_name =="refinenet":
    model =refinenet(input_shape=(256,256,5),num_classes=labels)
model.load_weights("h5File/unet_model_weight.h5")

#model.load_weights("h5File/"+model_name+'_model_weight.h5')
print("load model successfully")

x_img = tifffile.imread("/data/test_h/15out.tif")/255
ocr = np.zeros((x_img.shape[0]+235,x_img.shape[1]+277,7),'float16')
ocr[0:3093,0:3051,:]=x_img
ocr[3093:,3051,:]=0
tmp = np.zeros((x_img.shape[0]+235,x_img.shape[1]+277))
for i in range(int(ocr.shape[0]/128)-1):
    for j in range(int(ocr.shape[1]/128)-1):
        pred = model.predict(np.expand_dims(ocr[128*i:128*(i+1)+128,128*j:128*(j+1)+128,:],0))
        print(pred.shape)
        pred = np.squeeze(pred)
        tmp[128*i:128*(i+1)+128,128*j:128*(j+1)+128] = pred.argmax(axis=2)
print(np.unique(tmp))
rg =np.zeros((3093,3051))
rg = tmp[0:3093,0:3051]
tmpt = np.zeros((3093,3051,7))
for i in range(7):
    tmpt[:,:,i]=rg
tmpt[x_img==0] = 0
cv2.imwrite("data/pred.png",tmpt[:,:,0])
