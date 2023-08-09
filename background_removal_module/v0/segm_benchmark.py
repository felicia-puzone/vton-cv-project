import torch
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score
import random
from metrics import intersection_over_union
import logging
from PIL import Image
import os
import argparse
import json

""" 1- load img from folder 1
2 - load images from folder 2

foreach => IoI

3 - mean and variance
4 - log to file"""

class SegmentationTester():
    """
    Utility class to manage the testing accuracy score for some segmentation procedures
    """

    def __init__(self, trueMaskFolder: str, predictedMaskFolder: str):
        self._trueMaskFolder = trueMaskFolder
        self._predictedMaskFolder = predictedMaskFolder
        self._trueMasks = []
        self._predictedMasks = []

    def extractListImgs(self):

        path_true = self._trueMaskFolder
        path_predicted = self._predictedMaskFolder
        valid_images = [".png"]
        for f in os.listdir(path_true):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue

            image= cv2.imread(path_true + "\\" + f, cv2.IMREAD_GRAYSCALE)
            image = image == 255
            image = image.astype(int)

            self._trueMasks.append(image)

        for f in os.listdir(path_predicted):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue

            image= cv2.imread(path_predicted + "\\" + f, cv2.IMREAD_GRAYSCALE)
            image = image == 255
            image = image.astype(int)

            self._predictedMasks.append(image)

    def compute(self):
        scores = []
        for index, elem in enumerate(self._trueMasks):
            scores.append(intersection_over_union(torch.tensor(self._trueMasks[index]), torch.tensor(self._predictedMasks[index])))
        return scores

def get_args():
    parser = argparse.ArgumentParser(description='Perform Intersection over Union over two folders')
    parser.add_argument('--true_folder', '-t', metavar='TRUE', nargs='+', help='Folder name of the true masks', required=True)
    parser.add_argument('--predicted_folder', '-p', metavar='PRED', nargs='+', help='Folder name of the predicted masks')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(format='%(levelname)s %(asctime)s ==== %(message)s', filename='results.log', encoding='utf-8', level=logging.INFO)
    logging.info('started')

    tester = SegmentationTester(args.true_folder[0], args.predicted_folder[0])

    tester.extractListImgs()
    scores = tester.compute()

    mean = np.mean(scores)
    std = np.std(scores)

    logging.info('Scores:' + str(scores))

    logging.info('IoU mean Accuracy: ' + str(mean) + ', ' + 'Std: ' + str(std))

