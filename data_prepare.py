# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 下午12:55
# @Author  : Dongyang
# @File    : data_prepare.py
# @Software: PyCharm


import gdal
import numpy as np
from keras.utils import to_categorical

def max(a, b):
    return a if a >= b else b

def data_size(file_list):
    file_size=[]
    for file in file_list:
        dataset = gdal.Open(file)
        if dataset == None:
            print(file+"文件无法打开")

        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数
        im_bands = dataset.RasterCount #波段数

        file_size.append(im_width)
        file_size.append(im_height)

    imcolumns=max(file_size[0],file_size[2])
    imrows=max(file_size[1],file_size[3])

    return imcolumns,imrows,im_bands
#import tifffile as tiff
from skimage.external import tifffile as tiff
def read_data_array(file_name):
    dataset = tiff.imread(file_name)
    return dataset



file_list=['/home/dongyang/data/src_data_thailand/leftbottom.tif','/home/dongyang/data/src_data_thailand/righttop.tif']
mask_list = ['/home/dongyang/data/src_data_thailand/all_part1_mask_ENVI_class3.tif',
             '/home/dongyang/data/src_data_thailand/all_part2_mask_ENVI_class3.tif']

def read_data(file_list,mask_list):

    imcolumns,imrows,imbands=data_size(file_list)
    print(imcolumns,imrows,imbands)

    im_normal = np.zeros((imrows, imcolumns,imbands),dtype=np.float16)

    im_list = []
    ttone = np.zeros((2,imrows, imcolumns,imbands),dtype=np.float16)
    for k,files in enumerate(file_list):
        im_data = read_data_array(files)/255.0
        #im_data = np.transpose(im_data, (1,2, 0))
        #print(im_data.shape)
        #im_normal[0:im_data.shape[0], 0:im_data.shape[1],:] = im_data
        ttone[k,0:im_data.shape[0],0:im_data.shape[1],:]=im_data
        #im_normal_dim = np.expand_dims(im_normal, axis=0)#增加一个维度，连接时形成四维度数据结构
        #print(im_normal_dim.shape)
        #im_list.append(im_normal_dim)

    #im_concat = np.concatenate((im_list[0], im_list[1]), axis=0)
    # (2,3, 8181, 7362)
    #print(im_concat.shape)

    mask_normal_list = []
    ttlabel = np.zeros((2,imrows, imcolumns,imbands),dtype=np.uint8)
    for i,filet in enumerate(mask_list):
        mask_data = read_data_array(filet)
        y_img = np.squeeze(mask_data)
        l1 = (y_img ==1)
        l2 = (y_img ==2)
        background = np.logical_not(l1+l2)
        result_map = np.zeros((y_img.shape[0],y_img.shape[1],3))
        result_map[:, :, 0] = np.where(background, 1, 0)
        result_map[:, :, 1] = np.where(l1, 1, 0)
        result_map[:, :, 2] = np.where(l2, 1, 0)
        ttlabel[i,0:y_img.shape[0],0:y_img.shape[1],:]=result_map
        #mask_normal = np.zeros((imrows, imcolumns))

        #mask_normal[0:mask_data.shape[0], 0:mask_data.shape[1]] = mask_data
        # print(mask_normal.shape)
        #print(np.unique(mask_normal))
        #encoded = to_categorical(mask_normal,num_classes=3)
        #print(np.unique(encoded))
        #encoded=np.transpose(encoded, (2, 0, 1))
        #print(encoded.shape)
        #mask_normal_dim = np.expand_dims(tp, axis=0)#增加一个维度，连接时形成四维度数据结构
        #print(mask_normal_dim.shape)
        #mask_normal_list.append(mask_normal_dim)
        #print(encoded.shape)


    #mask_concat = np.concatenate((mask_normal_list[0], mask_normal_list[1]), axis=0)
    # (2,3, 8181, 7362)

    #print(mask_concat.shape)

    return ttone,ttlabel

if __name__ == '__main__':
    file_list = ['leftbottom.tif',
                 'righttop.tif']
    mask_list = ['all_part1_mask_ENVI_class3.tif',
                 'all_part2_mask_ENVI_class3.tif']

    im_concat, mask_concat=read_data(file_list, mask_list)
