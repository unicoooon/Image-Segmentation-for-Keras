# Image Segmentation with Keras : Implementation of Segnet, FCN, UNet,ICnet,PspNet,Deeplabv3,lstm,RefineNet.
## Models 

* FCN8
* Segnet 
* U-Net
* ICnet
* Pspnet
* Deeplabv3
* Lstm
* RefineNet

## Getting Started

### Prerequisites

* Keras
* Tensorflow

### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images 
* Annotations Folder - For the corresponding ground truth segmentation images



## Training the Model

To train the model run the following command:

```shell
python3 train.py --model model_names
```

Choose model_names from 'fcn', 'lstm','fcn32','unet', 'pspnet','deeplabv3p','deeplabv3','maskrcnn','segnet','refinenet','icnet'

## Getting the predictions

To get the predictions of a trained model

```shell
python3 test.py --model model_names
```
Choose model_names from 'fcn', 'lstm','fcn32','unet', 'pspnet','deeplabv3p','deeplabv3','maskrcnn','segnet','refinenet','icnet'

