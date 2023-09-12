import os
import shutil
import pickle
import copyreg
import cv2 as cv
import pre_processing as pre


def import_dataset(src_path, dst_path='./data/'):
    """
    usata per costruire il dataset con cui verrà costruita la repository
    chiamata dal main: import_dataset('D:\IngMagistrale\Computer_vision\project\dataset_example\images\\')
    :param src_path:
    :param dst_path:
    """
    img_names_list = os.listdir(src_path)

    for img_name in img_names_list:
        if img_name.find('_1') > -1:
            img = cv.imread(src_path + img_name, cv.IMREAD_GRAYSCALE)
            img = pre.perform_pre_processing(img)
            cv.imwrite(dst_path + img_name, img)


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


# costruzione della repository
def build_descriptors(img_path='./data/', filename='data_repo.pickle'):
    img_names_list = os.listdir(img_path)
    img_dict = {}

    orb = cv.ORB_create()

    for img_name in img_names_list:
        img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

        img = pre.perform_pre_processing(img)

        kp, des = orb.detectAndCompute(img, None)
        img_dict[img_name] = (kp, des)

    try:
        data_repo = open(filename, 'wb')
        pickle.dump(img_dict, data_repo)
        data_repo.close()
    except pickle.PickleError:
        print("Something went wrong during the img_dict dump!!!")


def repository_loader(filename='data_repo.pickle'):
    img_dict = None
    try:
        with open(filename, 'rb') as file:
            data_repo = open(filename, 'rb')
            img_dict = pickle.load(data_repo)
    except pickle.PickleError:
        print("Something went wrong during the img_dict load!!!")

    return img_dict


def build_repository(src_path, dst_path='./data/'):
    print("Checking destination path...")
    if len(os.listdir(dst_path)) < 1:
        print("Destination path NOT empty. Cleaning...")
        for file in os.scandir(dst_path):
            os.remove(file.path)

    import_dataset(src_path, dst_path)
    build_descriptors()

    print("Repository built!")


def build_images(repo_filepath='./data_repo.pickle', dst_path='./images/'):
    print("Checking destination path...")
    if len(os.listdir(dst_path)) < 1:
        print("Destination path NOT empty. Cleaning...")
        for file in os.scandir(dst_path):
            os.remove(file.path)

    print("Loading repository...")
    repo_dict = repository_loader(repo_filepath)

    print("Drawing keypoints...")
    for item_key, (item_kp, item_des) in repo_dict.items():
        img = cv.imread('./data/' + item_key, cv.IMREAD_GRAYSCALE)
        img = cv.drawKeypoints(img, item_kp, None, color=(0, 255, 0), flags=0)
        cv.imwrite(dst_path + item_key, img)

    print("All keypoints drawn. Job Done!")
