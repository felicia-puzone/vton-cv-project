import torch
import cv2
import os
import pycocotools.mask as mk
import numpy as np


# TODO: cfg.INPUT.MASK_FORMAT

def dress_code_loader(dir):
    os.listdir()

def bounding_box_extraction(mask:np.ndarray):
    mask[mask>0]=1
    a = np.where(mask != 0)
    print(a)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def mask_box_tensor(bbox,shape,image):
    print(bbox)
    final=torch.zeros(shape)
    final[bbox[0],bbox[2]:bbox[3]]=255
    final[bbox[1],bbox[2]:bbox[3]] = 255
    final[bbox[0]:bbox[1], bbox[2]] = 255
    final[bbox[0]:bbox[1],bbox[3]] = 255
    print(final)

    return final.numpy()+image

def tiktok_loader(dir):
    dataset = []
    for f_name in os.listdir(f"{dir}/images"):
        root = f"{dir}/images"
        f_dir = {
            "file_name": f"{root}/{f_name}",
            "sem_seg_file_name": f"{dir}/masks/{f_name}"
        }
        print(f_dir)
        dataset.append(f_dir)
    print(dataset)
    for image_dict in dataset:
        annotation_dic = {}
        mask = cv2.imread(image_dict["sem_seg_file_name"], flags=cv2.IMREAD_GRAYSCALE)
        image_dict["height"]=mask.shape[1]
        image_dict["width"] = mask.shape[0]
        image_dict["image_id"]=image_dict["sem_seg_file_name"].split("/")[-1].replace(".png","")
        mask[mask > 0] = 1
        annotation_dic["segmentation"]=mk.encode(np.asarray(mask, order="F"))

        image_dict["annotations"]=[annotation_dic]

if __name__ == "__main__":
    #tiktok_loader("../datasets/split_dataset/Validate")
    m=cv2.imread(f"../datasets/split_dataset/Validate/masks/136_00420.png",flags=cv2.IMREAD_GRAYSCALE)
    bbox=bounding_box_extraction(m)
    cancro=mask_box_tensor(bbox,m.shape,cv2.imread(f"../datasets/split_dataset/Validate/images/136_00420.png",cv2.IMREAD_GRAYSCALE))
    cv2.imshow("lool",cv2.imread(f"../datasets/split_dataset/Validate/images/136_00420.png",cv2.IMREAD_GRAYSCALE))
    cv2.waitKey()
    print(cancro)
    print(type(cancro))
    print(cancro.shape)
    cv2.imshow("lool",cancro)
    cv2.waitKey()