import csv
import cv2
from detectron2.structures.boxes import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

def load_train():
    # height,img_name,label,left,top,width,bottom,right
    label_dict = {}
    with open('/dataset/data.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for data in reader:
            img_name = data[1]
            if img_name in label_dict.keys():
                label_dict[img_name].append(data)
            else:
                label_dict[img_name] = [data]

    TRAIN_DATA_DIR = '/media/how/How/Class/deep_learning/cs-t0828-2020-hw2/dataset/train/'
    train_list = []
    all_data_list = []
    with open('/dataset/train.txt') as f:
        img_size_flie = open('/dataset/img_size.txt')
        for img_name in f:
            img_name = img_name[:-1]
            img = cv2.imread('/dataset/train/' + img_name)
            h, w = img_size_flie.readline().split()
            h = int(h)
            w = int(w)
            img_id = int(img_name.split('.')[0])
            ann_list_data = label_dict[img_name]
            ann_list = []
            for ann in ann_list_data:
                label = float(ann[2])
                left = float(ann[3])
                top = float(ann[4])
                right = float(ann[7])
                bottom = float(ann[6])
                bbox = [left, top, right, bottom]
                ann_dict = {'bbox': bbox, 'category_id': label, 'bbox_mode': BoxMode.XYXY_ABS}
                ann_list.append(ann_dict)
            
            img_data_dict = {'file_name':TRAIN_DATA_DIR + img_name,
                            'height': h, 'width': w,
                            'annotations': ann_list}
            all_data_list.append(img_data_dict)
    return all_data_list

DatasetCatalog.register("hw2_train", load_train)
MetadataCatalog.get("hw2_train").thing_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
