import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import image as img
from typing import overload
import argparse
import os
import shutil

class MaskFromLabelMap(nn.Module):
    """
    extremely simple class to extract the ground truth mask from the
    dataset for future testing, and perhaps as stand-in input for the segmentation

    """

    def __init__(self):
        super(MaskFromLabelMap, self).__init__()

    def forward(self, x: torch.Tensor):
        """

        :param x: Tensor, an extracted grayscale Tensor from the label_maps dataset or an image codename from dataset
        :return:
        """
        return torch.where(x == 0, x, 255)

def get_mask_from_label_map_file(label_dir:str,fileid:str):
    """
    mask extraction from ground truth label_map in functional form
    :param dataset_location: location of main directory of dataset
    :param type: type of clothing
    :param fileid: only the number_id section of the filename
    :return: original image Tensor and Tensor mask
    """
    segm_id = "_4.png"
    original_id="_0.jpg"
    im = cv2.imread(os.path.join(label_dir,f"{fileid}{segm_id}"), cv2.IMREAD_GRAYSCALE)
    im = torch.from_numpy(im)
    out = torch.where(im == 0, im, 255)
    im=cv2.imread(f"{label_dir}/{type}/images/{fileid}{original_id}")
    im = torch.from_numpy(im)
    return im, out

def get_mask_from_label_map(im:torch.Tensor):
    """
    mask extraction in functional form, this version requires the already opened
    ground truth label_map image as an input
    :param im: Label_map Tensor
    :return: Tensor mask
    """
    out = torch.where(im == 0, im, 255)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--label-dir",default=None)
    parser.add_argument("--mask-dir", default=None)
    opt = parser.parse_args()
    if opt.root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.root_dir) or not os.path.isdir(opt.root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)

    if opt.mask_dir is None:
        opt.mask_dir=os.path.join(opt.root_dir,"image-mask")
    if not os.path.exists(opt.mask_dir) or not os.path.isdir(opt.mask_dir):
        os.makedirs(opt.mask_dir)
    if opt.label_dir is None:
        opt.label_dir = os.path.join(opt.root_dir, "label_maps")
    if not os.path.exists(opt.label_dir) or not os.path.isdir(opt.label_dir):
        print(f"{opt.label_dir} directory does not exists")
        exit(-1)
    print(os.listdir(opt.mask_dir))
    image_dir = os.path.join(opt.root_dir, "images")


    if not os.path.exists(os.path.join(opt.root_dir,"cloth")) or not os.path.isdir(os.path.join(opt.root_dir,"cloth")):
        os.makedirs(os.path.join(opt.root_dir,"cloth"))
    for file in os.listdir(opt.label_dir):
        label_map=cv2.imread(os.path.join(opt.label_dir,file),cv2.IMREAD_GRAYSCALE)
        mask=get_mask_from_label_map(torch.from_numpy(label_map))
        image_id=file.split("_")[0]
        mask=mask.numpy()
        print(os.path.join(opt.mask_dir,f"{image_id}_0.png"))
        cv2.waitKey()
        cv2.imwrite(os.path.join(opt.mask_dir,f"{image_id}_0.png"),mask)

    for file in os.listdir(image_dir):
        filename_sections=file.split("_")
        print((file,filename_sections))
        if filename_sections[1]=="1.jpg":
            shutil.move(os.path.join(image_dir,file),os.path.join(opt.root_dir,"cloth",file))
