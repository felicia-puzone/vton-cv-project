import cv2
import torch
import time
import os
import shutil
import numpy as np
from multiprocessing import Process
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth


def image_resize(image, scale):
    im = cv2.imread(image)
    r = im.shape[0] / im.shape[1]
    new_dim = (int(im.shape[1] * scale), int(im.shape[0] * scale))
    resized = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)
    return resized


def dataset_resize(old_dir, new_dir, scale):
    try:
        os.makedirs(new_dir)
    except FileExistsError:
        pass
    for root, dirs, files in os.walk(old_dir):
        print(root)
        print(root.replace(old_dir, new_dir, 1))
        for name in files:
            if name.endswith((".png", ".jpg")):
                resized = image_resize(f"{root}\\{name}", scale)
                cv2.imwrite(root.replace(old_dir, new_dir, 1) + f"//{name}", resized)
            elif name.endswith(".npz"):
                continue
            else:
                shutil.copyfile(f"{root}\\{name}", root.replace(old_dir, new_dir, 1) + f"//{name}")
        print(dirs)
        for dir in dirs:
            print(dir)
            print(root.replace(old_dir, new_dir, 1))
            try:
                os.makedirs(root.replace(old_dir, new_dir, 1) + "/" + dir)
            except FileExistsError:
                continue


def dataset_upload(local_dir, gdrive_dir_id):
    """ dont use this it sucks, i should probs delete it altogether"""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    id_dict = {local_dir: gdrive_dir_id}
    print(id_dict)
    for root, dirs, files in os.walk(local_dir):
        print(root)
        for name in files:
            file_metadata = {'title': name,
                             "parent": [{'id': id_dict[root]}],
                             }
            print(file_metadata)
            f = drive.CreateFile(file_metadata)
            f.SetContentFile(f"{root}\\{name}")
            f.Upload()
            f = None
        for dir in dirs:
            file_metadata = {
                'title': dir,
                'parents': [{'id': id_dict[root]}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            print(file_metadata)
            f = drive.CreateFile(file_metadata)
            f.Upload()
            f = None
            files = drive.ListFile({'q': f"'{id_dict[root]}' in parents and trashed=false"}).GetList()
            fileID = None
            for file in files:
                if (file['title'] == dir):
                    fileID = file['id']
                    break
            if fileID is not None:
                id_dict[f"{root}\\{dir}"] = fileID


if __name__ == "__main__":
    # p1 = Process(target=dataset_resize,
    #            args=("D:\\Tumor\\DressCode\\dresses", "D:\\Tumor\\DressCodeResized\\dresses", 0.5), daemon=True)
    # p2 = Process(target=dataset_resize,
    #             args=("D:\\Tumor\\DressCode\\lower_body", "D:\\Tumor\\DressCodeResized\\lower_body", 0.5), daemon=True)
    # p3 = Process(target=dataset_resize,
    #             args=("D:\\Tumor\\DressCode\\upper_body", "D:\\Tumor\\DressCodeResized\\upper_body", 0.5), daemon=True)
    # p1.start(), p2.start(), p3.start()
    # image_resize("../../dataset_sample/dresses/images/020715_0.jpg",0.5)
    # data=np.load("../datasets/DressCodeResized/dresses/dense/020715_5_uv.npz")
    # lst=data.files
    # p1.join()
    # p2.join()
    # p3.join()
    dataset_upload("./trial_dir", "root")
