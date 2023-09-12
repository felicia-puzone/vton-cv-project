import torch
import cv2 as cv
from pathlib import Path
import numpy as np
import pickle
import copyreg

import config_file as cfg
import utils


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


def create_directory(dst_path: Path):
    dst_worn_cloth = dst_path / "worn-cloth"
    dst_in_shop_cloth = dst_path / "in-shop-cloth"

    if dst_in_shop_cloth.exists() and dst_worn_cloth.is_dir():
        print(f"{dst_in_shop_cloth} directory exists.")
    else:
        print(f"Did not find {dst_in_shop_cloth} directory, creating one...")
        dst_in_shop_cloth.mkdir(parents=True, exist_ok=True)

    if dst_worn_cloth.exists() and dst_worn_cloth.is_dir():
        print(f"{dst_worn_cloth} directory exists.")
    else:
        print(f"Did not find {dst_worn_cloth} directory, creating one...")
        dst_worn_cloth.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and dst_path.is_dir():
        print(f"{dst_path} directory exists.")
    else:
        print(f"Did not find {dst_path} directory, creating one...")
        dst_path.mkdir(parents=True, exist_ok=True)


def delete_diretory_content(target_path: Path):
    # target_dirlist = target_path.iterdir()
    # if any(target_dirlist):
    #     print(f"{target_path} is not empty. Deleting all files...")
    #     for item in target_dirlist:
    #         if item.is_file():
    #             item.unlink()

    if any(target_path.iterdir()):
        print(f"{target_path} is not empty. Deleting all files...")
        for item in target_path.iterdir():
            if item.is_file():
                item.unlink()


def delete_all_directories(target_path: Path = cfg.DST_DATA):
    in_shop_path = target_path / "in-shop-cloth"
    delete_diretory_content(target_path=in_shop_path)

    worn_path = target_path / "worn-cloth"
    delete_diretory_content(target_path=worn_path)

    data_path = target_path
    delete_diretory_content(target_path=data_path)


def get_in_shop_clothes(src_path: Path = cfg.DATASET_IMAGES_PATH, dst_path: Path = cfg.DST_IN_SHOP_CLOTH_PATH):
    clothes_path_list = list(src_path.glob("*_1.jpg"))
    for cloth_path in clothes_path_list:
        img = utils.open_and_resize_img(img_path=cloth_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)
        features = utils.get_features_from_image_rgb(img=img)

        cloth_name = cloth_path.name.split(sep="_")[0]
        img_path = dst_path / str(cloth_name + ".pickle")
        utils.save_features(data=features, file_str_path=img_path.as_posix())


def get_worn_clothes(src_path: Path = cfg.DATASET_IMAGES_PATH, dst_path: Path = cfg.DST_WORN_CLOTH_PATH,
                     label_maps_path=cfg.DATASET_LABEL_MAPS_PATH,
                     ):
    clothes_path_list = list(src_path.glob("*_0.jpg"))
    for cloth_path in clothes_path_list:
        img = utils.open_and_resize_img(img_path=cloth_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)

        cloth_name = cloth_path.name.split(sep="_")[0]
        label_map_path = label_maps_path / str(cloth_name + "_4.png")
        label_map = utils.open_and_resize_img(img_path=label_map_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)

        x, y, w, h = utils.find_bounding_box_from_mask(label_map=label_map, target_rgb_color=cfg.TARGET_RGB_COLOR)
        # N.B.: opencv image color image shape -> (Height, Width, Channels)
        worn_cloth = img[y:y + h, x:x + w, :]

        features = utils.get_features_from_image_rgb(img=worn_cloth)

        img_path = dst_path / str(cloth_name + ".pickle")
        utils.save_features(data=features, file_str_path=img_path.as_posix())


def write_train_test_files(target_path: Path, train_ratio: float = 0.8):
    """
        :param target_path: where train.txt and test.txt will be written
        :param train_ratio: split ratio in test and train set
        :return: None
    """
    cloth_path_list = list(Path(target_path / "in-shop-cloth").glob("*"))
    dataset_len = len(cloth_path_list)
    train_len = int(dataset_len * train_ratio)
    test_len = dataset_len - train_len
    count_train, count_test = (0, 0)

    with open((target_path / 'train.txt').as_posix(), 'w') as train_file, \
            open((target_path / 'test.txt').as_posix(), 'w') as test_file:
        for cloth_path in cloth_path_list:
            cloth_name = cloth_path.name
            prob = torch.rand(1).item()

            if prob <= 0.8:
                if count_train < train_len:
                    count_train += 1
                    print(cloth_name, file=train_file)
                elif count_test < test_len:
                    count_test += 1
                    print(cloth_name, file=test_file)
                else:
                    raise Exception("Something wrong in the train/test splitting!!!")
            else:
                if count_test < test_len:
                    count_test += 1
                    print(cloth_name, file=test_file)
                elif count_train < train_len:
                    count_train += 1
                    print(cloth_name, file=train_file)
                else:
                    raise Exception("Something wrong in the train/test splitting!!!")


if __name__ == "__main__":
    print(f"[INFO] Directory creation...")
    create_directory(dst_path=cfg.DST_DATA)

    delete_all_directories(target_path=cfg.DST_DATA)

    print(f"[INFO] Data processing (in-shop clothes)...")
    get_in_shop_clothes(src_path=cfg.DATASET_IMAGES_PATH, dst_path=cfg.DST_IN_SHOP_CLOTH_PATH)

    print(f"[INFO] Data processing (worn clothes)...")
    get_worn_clothes(src_path=cfg.DATASET_IMAGES_PATH, dst_path=cfg.DST_WORN_CLOTH_PATH,
                     label_maps_path=cfg.DATASET_LABEL_MAPS_PATH)

    print(f"[INFO] Writing train and test files...")
    write_train_test_files(target_path=cfg.DST_DATA, train_ratio=0.8)
