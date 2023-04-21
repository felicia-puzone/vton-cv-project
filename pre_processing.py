import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as nnf
from torchvision import models
import torchvision
import torchvision.transforms as T
import os
import pathlib
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pose_estimation(img):
    # img shape is [batch_size, channels, height, width]
    # qua si invocherebbe openpose/densepose per estrarre i keypoints
    # qui simulo la cosa con un po' di roba random
    batch_size = img.shape[0]
    height, width = 256, 192
    # key_channels è 18 come indicato nel paper dress-code
    key_channels = 18
    keypoints = torch.randint(0, 192, (batch_size, key_channels, 2)).to(device)

    ret_img = torch.zeros((batch_size, key_channels, height, width)).to(device)

    for batch_index in range(batch_size):
        for chn_index in range(key_channels):
            h, w = keypoints[batch_index, chn_index, :]
            ret_img[batch_index, chn_index, h, w] = 255

    # each keypoint is represented like a 3x3 white "rectangle", like in the dress-code paper
    kernel = torch.ones((3, 3), dtype=ret_img.dtype, device=ret_img.device, requires_grad=False)
    kernel = kernel[None, None, ...].repeat(ret_img.size(1), 1, 1, 1)
    ret_img = nnf.conv2d(ret_img, kernel, padding=1, groups=ret_img.size(1))

    return ret_img


def semantic_labeling(img):
    # img.shape is [batch_size, channels, height, width]
    # qua si invocherebbe densepose per eseguire la semantic segmentation (background e parti del corpo della persona)
    # qui simulo la cosa con un po' di roba random
    resize = T.Resize((256, 192))
    img = resize(img)

    batch_size, _, height, width = img.shape
    # label_channels è 25 come indicato nel paper dress-code
    label_channels = 25
    label_map = torch.zeros((batch_size, label_channels, height, width)).to(device)

    for batch_index in range(batch_size):
        for label_index in range(label_channels):
            # tmp is a random tensor with values at 0 or 1
            tmp = torch.randint(0, 2, (height, width))
            label_map[batch_index, label_index, :, :] = tmp

    return label_map


def masking_from_labeling(image, label_map):
    # label_map shape is [batch_size, label_channels, height, width]
    # il masking si basa sulla label_map ritornata da densepose
    batch_size, label_channels, height, width = label_map.shape

    # scelta arbitraria (senza alcun senso), giusto per simulare
    image[:, :, :, :] = torch.where(label_map[:, 0, :, :] != 0, image, 0)
    image[:, :, :, :] = torch.where(label_map[:, 1, :, :] != 0, image, 0)
    image[:, :, :, :] = torch.where(label_map[:, 2, :, :] != 0, image, 0)

    return image


if __name__ == '__main__':
    # !!! ora apro l'immagine direttamente dal codice, ma poi bisognerà automatizzare la cosa
    """
    assumo la seguente organizzazione delle directory:
    project_directory:
        |
        |--> data
        |        |--> person_image.jpg
        |--> pre_processing.py
    """
    data_dir = pathlib.Path("data")
    path = data_dir / "person_image.jpg"
    # image will have shape [image_channels, image_height, image_width]
    image = torchvision.io.read_image(str(path))
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    # image processing/filtering
    transforms = torch.nn.Sequential(
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        # resize con formato (H,W) in bassa risoluzione, come scritto in Dress-Code
        T.Resize((256, 192))
    )
    t_image = transforms(image)

    # person pose
    person_pose = pose_estimation(t_image)

    # semantic labels and masking
    label_map = semantic_labeling(t_image)
    masked_image = masking_from_labeling(t_image, label_map)

    # concatenation -> final person representation tensor
    person_representation = torch.cat((masked_image, person_pose), 1)
