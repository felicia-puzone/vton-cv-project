import torch
import cv2
import os
import pycocotools.mask as mk
import numpy as np
import json
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode


# TODO: cfg.INPUT.MASK_FORMAT


def bounding_box_extraction(mask: np.ndarray):
    mask[mask > 0] = 255
    a = np.where(mask != 0)
    print(mask.shape)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def class_bounding_box_extraction(mask: np.ndarray, class_id):
    a = np.where(mask == class_id)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def mask_box_tensor(bbox, shape, image):
    print(bbox)
    final = torch.zeros(shape)
    final[bbox[0], bbox[2]:bbox[3]] = 255
    final[bbox[1], bbox[2]:bbox[3]] = 255
    final[bbox[0]:bbox[1], bbox[2]] = 255
    final[bbox[0]:bbox[1], bbox[3]] = 255
    image_torch = torch.from_numpy(image)
    results = torch.where(final + image_torch > 255, 255, final + image_torch).to(torch.uint8)
    return results.numpy()


def bbox_json_export(data_dir):
    final_dic={}
    image_ids=[name.replace("_4.png","") for name in os.listdir(os.path.join(data_dir,"label_maps"))]
    for image_id in image_ids:
        return


def tiktok_loader(dir):
    # TODO actually finish this, its not at all useable
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
        image_dict["height"] = mask.shape[1]
        image_dict["width"] = mask.shape[0]
        image_dict["image_id"] = image_dict["sem_seg_file_name"].split("/")[-1].replace(".png", "")
        mask[mask > 0] = 1
        annotation_dic["segmentation"] = mk.encode(np.asarray(mask, order="F"))

        image_dict["annotations"] = [annotation_dic]


def dress_code_loader(img_dir,json_file):
    # TODO must be completely readapted to our dresscode dataset
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == "__main__":
    # tiktok_loader("../datasets/split_dataset/Validate")
    m = cv2.imread(f"../datasets/split_dataset/Validate/masks/136_00420.png", flags=cv2.IMREAD_GRAYSCALE)
    bbox = bounding_box_extraction(m)
    cancro = mask_box_tensor(bbox, m.shape, cv2.imread(f"../datasets/split_dataset/Validate/images/136_00420.png",
                                                       cv2.IMREAD_GRAYSCALE))
    cv2.imshow("lool", m)
    cv2.waitKey()
    print(cancro)
    print(type(cancro))
    print(cancro.shape)
    cv2.imshow("lool", cancro)
    cv2.waitKey()
