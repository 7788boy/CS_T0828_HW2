import cv2
from tqdm import tqdm
import json
import torch
from torchvision.transforms import ToTensor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

# Path
input_path = 'dataset/test/'
checkpoint_path = 'checkpoints/model_0244999.pth'

if __name__ == "__main__":
    cfg = get_cfg()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(checkpoint_path)
    model.eval()
    transform = ToTensor()
    result = []

    for index in tqdm(range(13068)):
        img_name = str(index + 1) + '.png'
        img = cv2.imread(input_path + img_name)
        img = transform(img)
        
        with torch.no_grad():
            predict = model([{'image':img[(2, 1, 0)]}])
        
        instance = predict[0]['instances']
        bboxes = instance.get_fields()['pred_boxes'].tensor
        scores = [int(s) for s in instance.get_fields()['scores']]
        labels = [int(s) for s in instance.get_fields()['pred_classes']]
        box_list = []
 
        for index, box in enumerate(bboxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            bbox = (y1, x1, y2, x2)
            box_list.append(bbox)
            
        result.append({'bbox': box_list, 'score': scores, 'label': labels})
        
    with open('result.json', 'w') as output_file:
        json.dump(result, output_file)
