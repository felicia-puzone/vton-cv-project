import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import os
from pathlib import Path
import pickle
import copyreg
import argparse

import config_file as cfg
import utils
import model_builder as mod


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


def prepare_plot(img, top_k_img):
    k = len(top_k_img)
    figure, ax = plt.subplots(nrows=1, ncols=k + 1, figsize=(10, 10))

    ax[0].imshow(img)
    ax[0].set_title("Image")

    for index in range(1, k + 1):
        ax[index].imshow(top_k_img[index - 1])
        ax[index].set_title(f"Top {index} retrieved image")

    figure.tight_layout()
    figure.show()
    plt.show()


def plot_img_list(img_list):
    k = len(img_list)
    figure, ax = plt.subplots(nrows=1, ncols=k + 1, figsize=(10, 10))

    for index in range(k):
        ax[index].imshow(img_list[index])
        ax[index].set_title(f"Top {index} retrieved image")

    figure.tight_layout()
    figure.show()
    plt.show()


def make_top_k_retrieval(model, worn_cloth_features, transform, k: int = 5, class_names=cfg.CLASS_NAMES,
                         repo_path: Path = cfg.DST_IN_SHOP_CLOTH_PATH, device="cuda"):
    in_shop_path_list = list(cfg.DST_IN_SHOP_CLOTH_PATH.glob("*"))

    model = model.to(device)
    model.eval()

    worn_tensor = transform(worn_cloth_features)

    match_list = []
    for in_shop_path in in_shop_path_list:
        in_shop_features = None
        try:
            with open(in_shop_path.as_posix(), 'rb') as in_shop_file:
                in_shop_features = pickle.load(in_shop_file)
        except pickle.PickleError:
            print(f"Something went wrong in the loading of {in_shop_path} data!!!")
        if in_shop_features is None:
            raise ValueError(f"From {in_shop_path} no loaded data!!!")

        in_shop_tensor = transform(in_shop_features)

        with torch.inference_mode():
            tensor = torch.cat((in_shop_tensor, worn_tensor), dim=0).unsqueeze(dim=0)
            tensor = tensor.to(torch.device(device=device))

            y_logits = model(tensor)
            pred_probs = torch.softmax(y_logits, dim=1)
            pred_label = torch.argmax(input=pred_probs, dim=1)

            if pred_label == 1:
                match_list.append((in_shop_path, pred_probs[:, pred_label]))

            print(f"In-shop cloth: {in_shop_path.name}"
                  f"| Predicted label: {class_names[pred_label.cpu()]} "
                  f"| Predicted probability: {pred_probs.cpu()}")

    print(f"Found {len(match_list)} possible matches.")
    top_k_match = sorted(match_list, reverse=True, key=lambda x: x[1])[:k]

    return top_k_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform a retrieval of top-k similar in-shop cloth items")
    parser.add_argument("--checkpoint_path", default=cfg.CHECKPOINT_PATH_DEFAULT, help="checkpoint file path")
    parser.add_argument("-image_path", help="reference image")
    parser.add_argument("-label_map_path", help="label map of the reference image")
    parser.add_argument("--top_k", default=cfg.TOP_K, help="top-k in-shop matches")
    parser.add_argument("--repo_path", default=cfg.DST_IN_SHOP_CLOTH_PATH,
                        help="repository of the in-shop cloth features")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    label_map_path = args.label_map_path
    top_k = args.top_k
    repo_path = args.repo_path

    model = mod.SimilarityNet(in_features=cfg.IN_FEATURES, hidden_units=cfg.HIDDEN_UNITS)
    utils.load_model(model=model, model_path=Path(checkpoint_path))
    model.to(cfg.DEVICE)

    img_path = Path(image_path)
    img = utils.open_and_resize_img(img_path=img_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)
    label_map = utils.open_and_resize_img(
        img_path=Path(label_map_path),
        new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)

    x, y, w, h = utils.find_bounding_box_from_mask(label_map=label_map, target_rgb_color=cfg.TARGET_RGB_COLOR)
    worn_cloth = img[y:y + h, x:x + w, :]
    worn_features = utils.get_features_from_image_rgb(worn_cloth)

    data_transform = utils.transform_to_tensor

    top_k_cloth = make_top_k_retrieval(model=model, worn_cloth_features=worn_features, transform=data_transform, k=top_k,
                                       class_names=cfg.CLASS_NAMES, repo_path=repo_path, device=cfg.DEVICE)

    with open('top_k_matches.txt', 'w') as out_file:
        if len(top_k_cloth) == 0:
            print("No match found!")
        else:
            print(top_k_cloth)
            top_k_img = []
            for cloth_name, cloth_prob in top_k_cloth:
                tmp_path = cfg.DATASET_IMAGES_PATH / str(cloth_name.name.split(".")[0] + "_1.jpg")
                tmp_img = utils.open_and_resize_img(img_path=tmp_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)
                top_k_img.append(tmp_img)
                print(cloth_name.name.split(".")[0], file=out_file)

            # invocare la funzione per la visualizzazione delle immagini
            img = utils.open_and_resize_img(img_path=img_path, new_dims=cfg.RESIZE_IMG_DIMS, is_grayscale=False)
            prepare_plot(img, top_k_img)
