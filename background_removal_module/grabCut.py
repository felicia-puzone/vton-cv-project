import torch
import numpy as np
import cv2
import random
from metrics import intersection_over_union
import logging
from PIL import Image
import os
import argparse




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

def saveImage(img, folder_path):
    cv2.imwrite(folder_path + "\\" + img["filename"], img["image"])

def applyTransform(listImgs : list, listImgsMask : list, outpath : str):

    """
        Applying our custom transform peforming
            - GrabCut
            - Median Blur Filter
        listImgs : a list of images
        listImgsMask : a list of the corresponding masks images with values in [0, 255]
    """

    transformedMasks = []

    output = {"filename": str,"image": np.array}

    for index, img in enumerate(listImgs):

        maskImg = listImgsMask[index]["image"]
        gcMask = maskImg
        gcMask[maskImg > 0] = cv2.GC_PR_FGD
        gcMask[maskImg == 0] = cv2.GC_BGD

        gcMask = gcMask.astype(np.uint8)
        print("[INFO] applying GrabCut.")
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        (gcMask, bgModel, fgModel) = cv2.grabCut(img["image"], gcMask,
                                                None, bgModel, fgModel, iterCount=2,
                                                mode=cv2.GC_INIT_WITH_MASK)
        outputMask = np.where((gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
        outputMask = (outputMask * 255).astype("uint8")

        outputMedianMask = cv2.medianBlur(outputMask, 13)


        output["filename"] = img["filename"]
        output["image"] = outputMedianMask

        saveImage(output, outpath)
        transformedMasks.append(output)

    return transformedMasks



def get_args():
    parser = argparse.ArgumentParser(description='Perform Intersection over Union over two folders')
    parser.add_argument('--folderImgs', '-i', metavar='img', nargs='+', help='Folder name of the true masks', required=True)
    parser.add_argument('--folderMasks', '-m', metavar='mask', nargs='+', help='Folder name of the true masks',
                        required=True)
    parser.add_argument('--out_folder', '-o', metavar='fold', nargs='+', help='Folder name of the predicted masks')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    list1, list2 = extractListImgs(args.folderImgs[0], args.folderMasks[0])

    out = applyTransform(list1, list2, args.out_folder[0])


    logging.basicConfig(format='%(levelname)s %(asctime)s ==== %(message)s', filename='results.log', encoding='utf-8', level=logging.INFO)
    logging.info('started')