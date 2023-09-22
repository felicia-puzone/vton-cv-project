import cv2
import torch
import time
import os
import shutil
import numpy as np
from multiprocessing import Process
import argparse

def image_resize(image, scale,interpolation):
    im = cv2.imread(image)
    r = im.shape[0] / im.shape[1]
    new_dim = (int(im.shape[1] * scale), int(im.shape[0] * scale))
    resized = cv2.resize(im, new_dim, interpolation=interpolation)
    return resized

def get_mask_from_label_map(im:torch.Tensor):
    """
    mask extraction in functional form, this version requires the already opened
    ground truth label_map image as an input
    :param im: Label_map Tensor
    :return: Tensor mask
    """
    out = torch.where(im == 0, im, 255)
    return out

def derivative_directories(opt,cloth_type):

    dataset_resize(os.path.join(opt.old_root_dir,cloth_type),os.path.join(opt.new_root_dir,cloth_type), 0.5)
    root_dir=opt.new_root_dir
    image_dir = os.path.join(root_dir, "images")
    label_dir=os.path.join(root_dir, "label-maps")
    cloth_dir = os.path.join(root_dir, "cloth")
    image_mask_dir= os.path.join(root_dir, "image-mask")
    if not os.path.exists(os.path.join(root_dir, "cloth")) or not os.path.isdir(
            os.path.join(root_dir, "cloth")):
        os.makedirs(os.path.join(root_dir, "cloth"))
    for file in os.listdir(label_dir):
        label_map = cv2.imread(os.path.join(label_dir, file), cv2.IMREAD_GRAYSCALE)
        mask = get_mask_from_label_map(torch.from_numpy(label_map))
        image_id = file.split("_")[0]
        mask = mask.numpy()
        print(os.path.join(image_mask_dir, f"{image_id}_0.png"))
        cv2.imwrite(os.path.join(image_mask_dir, f"{image_id}_0.png"), mask)

    for file in os.listdir(image_dir):
        filename_sections = file.split("_")
        print((file, filename_sections))
        if filename_sections[1] == "1.jpg":
            shutil.move(os.path.join(image_dir, file), os.path.join(root_dir, "cloth", file))

def dataset_resize(old_dir, new_dir, scale):
    try:
        os.makedirs(new_dir)
    except FileExistsError:
        pass
    for root, dirs, files in os.walk(old_dir):
        print(root)
        print(root.replace(old_dir, new_dir, 1))
        for name in files:
            if name.endswith(".jpg"):
                resized = image_resize(f"{root}/{name}", scale,interpolation=cv2.INTER_AREA)
                cv2.imwrite(root.replace(old_dir, new_dir, 1) + f"/{name}", resized)
            elif name.endswith(".png"):
                resized = image_resize(f"{root}/{name}", scale,interpolation=cv2.INTER_NEAREST_EXACT)
                cv2.imwrite(root.replace(old_dir, new_dir, 1) + f"/{name}", resized)
            elif name.endswith(".npz"):
                continue
            else:
                shutil.copyfile(f"{root}/{name}", root.replace(old_dir, new_dir, 1) + f"/{name}")
        print(dirs)
        for dir in dirs:
            print(dir)
            print(root.replace(old_dir, new_dir, 1))
            try:
                os.makedirs(root.replace(old_dir, new_dir, 1) + "/" + dir)
            except FileExistsError:
                continue




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root-dir", default=None)
    parser.add_argument("--new-root-dir", default=None)
    opt = parser.parse_args()
    if opt.old_root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.old_root_dir) or not os.path.isdir(opt.old_root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)
    if opt.new_root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.new_root_dir) or not os.path.isdir(opt.new_root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)

    p1 = Process(target=dataset_resize,

                 args=(os.path.join(opt.old_root_dir,"dresses"),os.path.join(opt.new_root_dir,"dresses"), 0.5), daemon=True)
    p2 = Process(target=dataset_resize,
                 args=(os.path.join(opt.old_root_dir,"upper_body"),os.path.join(opt.new_root_dir,"upper_body"), 0.5), daemon=True)
    p3 = Process(target=dataset_resize,
                 args=(os.path.join(opt.old_root_dir,"lower_body"),os.path.join(opt.new_root_dir,"lower_body"), 0.5), daemon=True)

    p1.start(), p2.start(), p3.start()
    p1.join()
    p2.join()
    p3.join()
