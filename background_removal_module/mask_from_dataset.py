import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import image as img
from typing import overload

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

def get_mask_from_label_map_file(dataset_location:str, type:str, fileid:str):
    """
    mask extraction from ground truth label_map in functional form
    :param dataset_location: location of main directory of dataset
    :param type: type of clothing
    :param fileid: only the number_id section of the filename
    :return: original image Tensor and Tensor mask
    """
    segm_id = "_4.png"
    original_id="_0.jpg"
    im = cv2.imread(f"{dataset_location}/{type}/label_maps/{fileid}{segm_id}", cv2.IMREAD_GRAYSCALE)
    im = torch.from_numpy(im)
    out = torch.where(im == 0, im, 255)
    im=cv2.imread(f"{dataset_location}/{type}/images/{fileid}{original_id}")
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
    file = "020715"
    im,out = get_mask_from_label_map_file(f"../../../dataset_sample", "dresses", file)
    plt.subplot(121), plt.imshow(cv2.cvtColor(im.numpy(), cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(out)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
