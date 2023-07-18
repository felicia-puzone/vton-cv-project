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


def generateMask(img, sharpenLevel = 0):

    #ADJUSTING BRIGHTNESS/CONTRAST

    # define the contrast and brightness value
    contrast = 1.05    # Contrast control ( 0 to 127)
    brightness = 0.05  # Brightness control (0-100)

    # call addWeighted function. use beta = 0 to effectively only

    adjusted = cv2.addWeighted(img, contrast, img, 0, brightness)

    #Sharpening kernel 1
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if(sharpenLevel == 1):
        adjusted = cv2.filter2D(adjusted, -1, kernel)

    v = np.median(adjusted)
    sigma = 0.99

    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(adjusted, 0, 255)

    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    contours, hierarchy = cv2.findContours(edges,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Canny Edges After Contouring

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    mask = np.zeros(img_input.shape, dtype=np.uint8)

    cv2.drawContours(mask, contours, 0, color=(255, 255, 255), thickness=cv2.FILLED)

    if (sharpenLevel == 1):
        mask = cv2.medianBlur(mask, 33)

    #computing White/Black Ratio

    mask_white_count = (mask==255).sum()
    mask_black_count = (mask==0).sum()

    wb_ratio = mask_white_count/mask_black_count

    return mask, wb_ratio


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

    first_fixed_count = 0
    total_num_of_images = 0
    total_of_good_masks = 0

    for file in os.listdir(cloth_dir):
        img_input = cv2.imread(os.path.join(cloth_dir, file), cv2.IMREAD_GRAYSCALE)
        total_num_of_images += 1

        mask, wb_ratio = generateMask(img_input, sharpenLevel=0)

        if wb_ratio >= 0.1: total_of_good_masks += 1

        if wb_ratio < 0.1:
            print("Failed to draw mask at first try. Adding sharpening kernel")

            mask, wb_ratio = generateMask(img_input, sharpenLevel=1)

            if wb_ratio >= 0.1:
                first_fixed_count +=1
                total_of_good_masks += 1

        cv2.imwrite(os.path.join(opt.root_dir,"cloth-mask",file), mask)

    print("Finished.")
    print("Number of first pass fixed images:", first_fixed_count)
    print("Total images:", total_num_of_images)
    print("Total of good masks:", total_of_good_masks)

