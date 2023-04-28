import torch
import numpy as np
import cv2
from sklearn.metrics import jaccard_score,f1_score
import random

def intersection_over_union(true:torch.Tensor,pred:torch.Tensor):
    if len(true.shape)!=2:
        print(f"true Tensor isn't expected 2D mask")
        return
    if len(pred.shape) != 2:
        print(f"pred Tensor isn't expected 2D mask")
        return
    unrolled_true=torch.flatten(true)
    unrolled_pred = torch.flatten(pred)
    score= jaccard_score(unrolled_true,unrolled_pred)
    return score
def dice(true:torch.Tensor,pred:torch.Tensor):
    if len(true.shape) != 2:
        print(f"true Tensor isn't expected 2D mask")
        return
    if len(pred.shape) != 2:
        print(f"pred Tensor isn't expected 2D mask")
        return
    unrolled_true=torch.flatten(true)
    unrolled_pred = torch.flatten(pred)
    score= f1_score(unrolled_true,unrolled_pred)
    return score

if __name__=="__main__":
    true=torch.rand((100,100))
    pred=true>0.6
    true=true>0.5
    print(dice(true,pred))
    print(intersection_over_union(true,pred))