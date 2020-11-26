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
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
* OpenCV is optional and needed by demo and visualization
* Numpy 1.19.2
* Tqdm 4.51.0
* Cuda 10.1
* Detectron 2
 
You need to create a dictory names 'checkpoints' to save chekpoint.  
Download Detectron2 by follow the github  
https://github.com/facebookresearch/detectron2
  
## Usage
Run train.py to start training.   
```
python train.py
```

Test the model  
```
python Test.py
```

## Reference
https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf
http://ufldl.stanford.edu/housenumbers/
https://github.com/facebookresearch/detectron2

###### tags: `Object detection` `Deep learning` `NCTU CS` `Faster-RCNN` `Detectron`
