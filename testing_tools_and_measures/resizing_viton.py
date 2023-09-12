import cv2
import numpy as np
import os

up_width = 384
up_height = 512
up_points = (up_width, up_height)

infolder = r'C:\Users\ruteryan\Desktop\cp-vton-plus\data\test\image-parse-new'
outfolder = r'C:\Users\ruteryan\Desktop\viton_resized\image-parse-new'


for filename in os.listdir(infolder):
    img = cv2.imread(os.path.join(infolder,filename))
    if img is not None:
        img_resized = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outfolder,filename), img_resized)
