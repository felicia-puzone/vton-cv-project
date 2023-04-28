import random
import os
import shutil
from multiprocessing import Process


def get_training_validating_testing_sets(dataset_dir, train, test, valid):
    data = os.listdir(f"{dataset_dir}\\images")
    random.shuffle(data)
    num = len(data)
    train_num = int(num * train)
    test_num = int(num * test)
    valid_num = int(num * valid)
    train_set = data[0:train_num]
    test_set = data[train_num:train_num + test_num]
    validate_set = data[test_num + train_num:]

    return train_set, test_set, validate_set


def make_training_test_validation_dirs(original_dataset_dir, new_dir_location, train, test, valid):
    new_test_path = f'{new_dir_location}\\Test'
    new_train_path = f'{new_dir_location}\\Train'
    new_validate_path = f'{new_dir_location}\\Validate'
    for dir in [new_train_path, new_test_path, new_validate_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

        os.makedirs(f"{dir}\\images")
        os.makedirs(f"{dir}\\masks")
    train_set, test_set, validate_set = get_training_validating_testing_sets(original_dataset_dir, train, test, valid)
    for image in train_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_train_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_train_path}\\masks\\{image}")
    for image in test_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_test_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_test_path}\\masks\\{image}")

    for image in validate_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_validate_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_validate_path}\\masks\\{image}")


def section_selection_copy(section_dir, new_section_dir, selection_list):
    for root, dirs, files in os.walk(section_dir):
        print(root)
        for file in files:
            if file.endswith("txt"):
                shutil.copy(f"{root}\\{file}",f"{root.replace(section_dir, new_section_dir, 1)}\\{file}")
            elif file.split("_")[0] in selection_list:
                shutil.copy(f"{root}\\{file}",f"{root.replace(section_dir, new_section_dir, 1)}\\{file}")
        for dir in dirs:
            try:
                os.makedirs(f"{root.replace(section_dir, new_section_dir, 1)}\\{dir}")
            except FileExistsError:
                continue


def file_select(main_dir, ratio):
    image_id_dict = {}
    for section in ["dresses", "lower_body", "upper_body"]:
        data = os.listdir(f"{main_dir}\\{section}\\images")
        id_list = [s.replace("_0.jpg", "") for s in data if s.endswith("0.jpg")]
        random.shuffle(id_list)
        num = len(id_list)
        new_num = int(ratio * num)
        image_id_dict[section] = id_list[:new_num]
    return image_id_dict


def get_reduced_dress_dataset(data_dir, new_data_dir, ratio):
    selection_dict = file_select(data_dir, ratio)

    for file in os.listdir(data_dir):
        if file in ["dresses", "lower_body", "upper_body"]:
            try:
                os.makedirs(f"{new_data_dir}\\{file}")
            except FileExistsError:
                continue
        else:
            shutil.copy(f"{data_dir}\\{file}", f"{new_data_dir}\\{file}")
    p1 = Process(target=section_selection_copy,
                 args=(f"{data_dir}\\dresses", f"{new_data_dir}\\dresses", selection_dict["dresses"]), daemon=True)
    p2 = Process(target=section_selection_copy,
                 args=(f"{data_dir}\\upper_body", f"{new_data_dir}\\upper_body", selection_dict["upper_body"]),
                 daemon=True)
    p3 = Process(target=section_selection_copy,
                 args=(f"{data_dir}\\lower_body", f"{new_data_dir}\\lower_body", selection_dict["lower_body"]),
                 daemon=True)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    return


if __name__ == "__main__":
    # train_set, test_set, validate_set=get_training_validating_testing_sets("<insert_image_directory_here>",0.5,0.25,0.25)
    # make_training_test_validation_dirs("F:\pycharm_projects_F\CV_P_13_2\dataset_division_test",".",0.5,0.25,0.25)
    get_reduced_dress_dataset("D:\\Tumor\\DressCodeResized","D:\\Tumor\\DressCodeDoubleResized",0.5)
