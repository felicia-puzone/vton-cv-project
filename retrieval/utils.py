import cv2 as cv
import numpy as np
import pickle
import copyreg
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
from pathlib import Path

import config_file as cfg


def open_and_resize_img(img_path, new_dims, is_grayscale: bool = True):
    if is_grayscale:
        img = cv.imread(img_path.as_posix(), cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(img_path.as_posix(), cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, new_dims)

    if img is None:
        raise ValueError(f"Image {img_path} not correctly loaded!")

    return img


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


def save_features(data, file_str_path: str):
    with open(file_str_path, 'wb') as file:
        pickle.dump(data, file)


def get_features(img):
    orb = cv.ORB_create()
    img_kp, img_des = orb.detectAndCompute(img, None)

    sorted_indices = sorted(range(len(img_kp)), key=lambda i: img_kp[i].response, reverse=True)
    best_indices = sorted_indices[:cfg.N_BEST_KEYPOINTS]

    features = np.zeros(cfg.N_FEATURES)
    n_features = len(best_indices) * (2 + 1 + 1 + cfg.N_MAX_ORB_POINT_FEATURES)
    if n_features == 0:
        return features

    index = 0
    for i in best_indices:
        kp = img_kp[i]
        des = img_des[i]
        px, py = kp.pt
        features[index:index + 4] = [px, py, kp.angle, kp.response]
        index += 4
        features[index:index+cfg.N_MAX_ORB_POINT_FEATURES] = des
        index += cfg.N_MAX_ORB_POINT_FEATURES

    index = cfg.N_FEATURES - cfg.N_HISTOGRAM_BINS
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    features[index:] = np.reshape(hist, (-1,))

    return features


def get_features_from_image_rgb(img):
    n_channels = img.shape[-1]
    features_list = []
    for channel in range(n_channels):
        features_list.append(get_features(img[:, :, channel]))

    features = np.concatenate(features_list)

    return features


def get_features_from_image(img):
    orb = cv.ORB_create()
    img_kp, img_des = orb.detectAndCompute(img, None)

    sorted_indices = sorted(range(len(img_kp)), key=lambda i: img_kp[i].response, reverse=True)
    best_indices = sorted_indices[:cfg.N_BEST_KEYPOINTS]

    features = np.zeros(cfg.N_FEATURES)
    n_features = len(best_indices) * (2 + 1 + 1 + cfg.N_MAX_ORB_POINT_FEATURES)
    if n_features == 0:
        return features

    index = 0
    for i in best_indices:
        kp = img_kp[i]
        des = img_des[i]
        px, py = kp.pt
        features[index:index + 4] = [px, py, kp.angle, kp.response]
        index += 4
        features[index:index+cfg.N_MAX_ORB_POINT_FEATURES] = des
        index += cfg.N_MAX_ORB_POINT_FEATURES

    index = cfg.N_FEATURES - cfg.N_HISTOGRAM_BINS
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    features[index:] = np.reshape(hist, (-1,))

    return features


def find_bounding_box_from_mask(label_map, target_rgb_color):
    mask = np.all(label_map == target_rgb_color, axis=-1)
    gray_mask = np.zeros_like(label_map)
    gray_mask[mask] = [255, 255, 255]
    gray_mask = cv.cvtColor(gray_mask, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(gray_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)

    return x, y, w, h


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load_model(model, model_path: str):
    model.load_state_dict(torch.load(model_path))


def transform_to_tensor(src_data):
    return torch.tensor(src_data, dtype=torch.float32)


