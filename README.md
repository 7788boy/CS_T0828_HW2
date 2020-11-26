# Project: Object detection

## Overview
The target of this homework is object detection and recognize numbers.  
Training 11185 pictures with cars in different type and 196 labels.   
I use Detectron and Fast RCNN as pretrained model.  

## Hardware
The following specs were used to create the original solution.

* Ubuntu 18.04 LTS
* NVIDIA GeForce RTX 2080

## Download Official Image
You can download the training and testing dataset from Google Drive.  
https://drive.google.com/drive/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl  

## Installation
* Pytorch 1.7.0
* Numpy 1.19.2
* Torchvision 0.8.1
* Tensorboard 2.3.0
* Tqdm 4.51.0
* Cuda 10.1
* Detectron
* Fast RCNN  
```
pip install keras_efficientnets
pip install efficientnet_pytorch
```   
You need to create a dictory names 'checkpoints' to save chekpoint.  
  
## Usage
Run train.py to start training.   
```
python train.py
```
You can set dirfferent parameter by the following command.   
```
--h
--mode
--epochs
--batch
--ckptID
```

Test the model  
```
python train.py --mode test --ckptID <your checkpoint ID>
```

Check training log  
```
tensorboard --logdir=runs
```
## Reference
https://arxiv.org/abs/1905.11946  
https://zhuanlan.zhihu.com/p/102467338  
https://pytorch.org/docs/stable/torchvision/models.html#classification  
https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/  

###### tags: `Image classification` `Deep learning` `NCTU CS` `EfficientNet`
