from tqdm import tqdm
import cv2

with open('train.txt') as f:
    img_size = open('img_size.txt', 'w')
    for img_name in tqdm(f):
        img_name = img_name[:-1]
        img = cv2.imread('train/' + img_name)
        h, w, _ = img.shape
            
        img_size.write('%d %d\n' % (h, w))

