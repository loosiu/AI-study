from PIL import Image
import os
import cv2
import numpy as np

for i in range(1,13):
    path = f'D:/AI dataset/ECG2scalogram_oneImg_trValTst/lead_{i}/val/SR'        
    globals()[f'SR_image_lead{i}'] = os.listdir(path)
    
   
for num in range(357):
    images = []
    for i in range(1,13):
        image = globals()[f'SR_image_lead{i}'][num]
        save_path = f'D:/AI dataset/ECG2scalogram_oneImg_trValTst/merge/val/SR/{image}'
        lead_path = f'D:/AI dataset/ECG2scalogram_oneImg_trValTst/lead_{i}/val/SR'
        image_path = f'{lead_path}/{image}'
        img = cv2.imread(image_path)
        images.append(img)
    concat_image = np.vstack(images)
    cv2.imwrite(save_path, concat_image)