import torch
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score
import random
from statistics import median, mean
import argparse
import os
import shutil
import json


def intersection_over_union(true: torch.Tensor, pred: torch.Tensor):
    if len(true.shape) != 2:
        print(f"true Tensor isn't expected 2D mask")
        return
    if len(pred.shape) != 2:
        print(f"pred Tensor isn't expected 2D mask")
        return
    unrolled_true = torch.flatten(true)
    unrolled_pred = torch.flatten(pred)
    score = jaccard_score(unrolled_true, unrolled_pred)
    return score


def dice(true: torch.Tensor, pred: torch.Tensor):
    if len(true.shape) != 2:
        print(f"true Tensor isn't expected 2D mask")
        return
    if len(pred.shape) != 2:
        print(f"pred Tensor isn't expected 2D mask")
        return
    unrolled_true = torch.flatten(true)
    unrolled_pred = torch.flatten(pred)
    score = f1_score(unrolled_true, unrolled_pred)
    return score


def metric_results(true_dir, pred_dir):
    scores = {}
    iou_list = []
    dice_list = []
    for image_file in os.listdir(true_dir):
        print(f"processing image {image_file}")
        true_tensor = torch.from_numpy(cv2.imread(os.path.join(true_dir, image_file), cv2.IMREAD_GRAYSCALE))
        true_tensor = true_tensor == 255
        try:
            pred_tensor = torch.from_numpy(cv2.imread(os.path.join(pred_dir, image_file), cv2.IMREAD_GRAYSCALE))
            pred_tensor = pred_tensor == 255
        except KeyError:
            print(f"ground-truth dir does not possess {image_file} mask")
            return None
        scores[image_file] = {
            "dice": dice(true_tensor, pred_tensor),
            "iou": intersection_over_union(true_tensor, pred_tensor)
        }
        iou_list.append(scores[image_file]["iou"])
        dice_list.append(scores[image_file]["dice"])
    scores["iou_list"] = iou_list
    scores["dice_list"] = dice_list
    scores["iou_median"] = median(iou_list)
    scores["dice_median"] = median(dice_list)
    scores["iou_mean"] = mean(iou_list)
    scores["dice_mean"] = mean(dice_list)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluates the model outputs by using the dice and score metrics")
    parser.add_argument("-t", help="directory where the ground-truth masks reside", dest="true_dir", required=True,
                        nargs='?')
    parser.add_argument("-p", help="directory where the predicted masks reside", dest="pred_dir", required=True,
                        nargs='?')
    parser.add_argument("-n", help="number of masks loaded at once (not yet implemented)", default=10, dest="batch",
                        nargs='?')
    parser.add_argument("-o", help="name and location of json output file", default=10, dest="json_loc", nargs='?')
    args = parser.parse_args()
    pred_dir = args.pred_dir
    true_dir = args.true_dir
    json_path = args.json_loc
    batch = args.batch
    if not os.path.isdir(pred_dir):
        print(f"{pred_dir} is not a directory")
        exit(-1)
    if not os.path.isdir(true_dir):
        print(f"{true_dir} is not a directory")
        exit(-1)
    scores = metric_results(true_dir, pred_dir)
    json.dump(scores, open(json_path, "w"), indent=6)
