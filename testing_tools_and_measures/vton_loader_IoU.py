import torch
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score
import random
import logging
from PIL import Image
import os
import argparse
import json


def intersection_over_union(true, pred):
    if len(true.shape) != 2:
        print(f"true Tensor isn't expected 2D mask")
        return
    if len(pred.shape) != 2:
        print(f"pred Tensor isn't expected 2D mask")
        return

    img_true = np.array(true).ravel()
    img_pred = np.array(pred).ravel()

    score = jaccard_score(img_true, img_pred)
    return score


def extraxt_list_imgs(path_true_masks, path_warped_masks, namelist):

    true_masks_list = []
    warped_masks_list = []

    '''true masks'''
    for name in namelist:
        filename = name + '_4.png'
        print('Loading mask %s' % filename)
        im_parse = cv2.imread(path_true_masks + "\\" + filename, cv2.IMREAD_GRAYSCALE)
        parse_array = np.array(im_parse)
        image = (parse_array == 14).astype(np.float32) + \
            (parse_array == 128).astype(np.float32)    # upper-clothes labels
        image = image != 0
        image = image.astype(int)

        true_masks_list.append(image)

    '''warped masks'''
    for name in namelist:
        filename = name + '_0.jpg'
        print('Loading mask %s' % filename)
        image = cv2.imread(path_warped_masks + "\\" + filename, cv2.IMREAD_GRAYSCALE)
        image = image == 255
        image = image.astype(int)

        warped_masks_list.append(image)

    return true_masks_list, warped_masks_list


def get_args():
    parser = argparse.ArgumentParser(
        description='Perform Intersection over Union over two folders and a txt list mapper')
    parser.add_argument('--log_info', help='Description of the log details (testing settings)',
                        required=True)
    parser.add_argument('--true_folder', '-t', metavar='TRUE', nargs='+', help='Folder name of the true masks',
                        required=True)
    parser.add_argument('--predicted_folder', '-p', metavar='PRED', nargs='+',
                        help='Folder name of the predicted masks')
    parser.add_argument('--datalist', '-l', metavar='LIST', nargs='+', help='List of images (train/test division')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # load data list
    namelist = []
    with open(args.datalist[0], 'r') as f:
        for line in f.readlines():
            im_name = (line.strip().split()[0]).split('_')[0]
            namelist.append(im_name)

    true_masks, warped_masks = extraxt_list_imgs(args.true_folder[0], args.predicted_folder[0], namelist)

    print('Extracted mask lists')

    scores = []
    for index, elem in enumerate(true_masks):
        score = intersection_over_union(true_masks[index], warped_masks[index])
        print('Score %d: %.3f' % (index, score))
        scores.append(score)


    logging.basicConfig(format='%(levelname)s %(asctime)s ==== %(message)s', filename='results.log', encoding='utf-8',
                        level=logging.INFO)
    logging.info(args.log_info)

    mean = np.mean(scores)
    std = np.std(scores)

    #logging.info('Scores:' + str(scores))

    logging.info('IoU mean Accuracy: ' + str(mean) + ', ' + 'Std: ' + str(std))
