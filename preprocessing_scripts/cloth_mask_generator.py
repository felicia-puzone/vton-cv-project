import cv2
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from typing import overload
import argparse
import os
import shutil

def extractListImgs(inFolderImgs : str, inFolderMasks : str):

    img = {"filename": str,"image": np.array}
    listImgs = []
    listMasks = []

    valid_images = [".png"]
    for f in os.listdir(inFolderImgs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        image= cv2.imread(inFolderImgs + "\\" + f, cv2.IMREAD_COLOR)

        listImgs.append({"filename": f,"image":image})

    for f in os.listdir(inFolderMasks):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        mask = cv2.imread(inFolderMasks + "\\" + f, cv2.IMREAD_GRAYSCALE)
        mask = mask == 255
        mask = mask.astype(int)

        listMasks.append({"filename": f,"image":mask})

    return listImgs, listMasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=None)

    opt = parser.parse_args()


    if opt.root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.root_dir) or not os.path.isdir(opt.root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)


    if not os.path.exists(os.path.join(opt.root_dir,"cloth-mask")) or not os.path.isdir(os.path.join(opt.root_dir,"cloth-mask")):
        os.makedirs(os.path.join(opt.root_dir,"cloth-mask"))

    cloth_dir = os.path.join(opt.root_dir,"cloth")

    for file in os.listdir(cloth_dir):
        img_input = cv2.imread(os.path.join(cloth_dir, file), cv2.IMREAD_GRAYSCALE)

        v = np.median(img_input)
        sigma = 0.99

        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))

        edges = cv2.Canny(img_input, 0, 255)

        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

        contours, hierarchy = cv2.findContours(edges,
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #Canny Edges After Contouring

        print("Number of Contours found = " + str(len(contours)))

        # Draw all contours
        # -1 signifies drawing all contours
        mask = np.zeros(img_input.shape, dtype=np.uint8)

        cv2.drawContours(mask, contours, 0, color=(255, 255, 255), thickness=cv2.FILLED)

        cv2.imwrite(os.path.join(opt.root_dir,"cloth-mask",file), mask)

