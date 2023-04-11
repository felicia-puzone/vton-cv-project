
import random
import os
import shutil

def get_training_validating_testing_sets(dataset_dir,train,test,valid):
    data=os.listdir(f"{dataset_dir}\\images")
    random.shuffle(data)
    num=len(data)
    train_num=int(num*train)
    test_num = int(num * test)
    valid_num = int(num * valid)

    train_set=data[0:train_num]
    test_set=data[train_num:train_num+test_num]
    validate_set=data[test_num+train_num:]

    return train_set,test_set,validate_set

def make_training_test_validation_dirs(original_dataset_dir,new_dir_location,train,test,valid):
    new_test_path = f'{new_dir_location}\\Test'
    new_train_path =f'{new_dir_location}\\Train'
    new_validate_path = f'{new_dir_location}\\Validate'
    for dir in [new_train_path,new_test_path,new_validate_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

        os.makedirs(f"{dir}\\images")
        os.makedirs(f"{dir}\\masks")
    train_set, test_set, validate_set=get_training_validating_testing_sets(original_dataset_dir,train,test,valid)
    for image in train_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_train_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_train_path}\\masks\\{image}")
    for image in test_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_test_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_test_path}\\masks\\{image}")

    for image in validate_set:
        shutil.copy(f"{original_dataset_dir}\\images\\{image}", f"{new_validate_path}\\images\\{image}")
        shutil.copy(f"{original_dataset_dir}\\masks\\{image}", f"{new_validate_path}\\masks\\{image}")


if __name__=="__main__":
    #train_set, test_set, validate_set=get_training_validating_testing_sets("<insert_image_directory_here>",0.5,0.25,0.25)
    make_training_test_validation_dirs("F:\pycharm_projects_F\CV_P_13_2\dataset_division_test",".",0.5,0.25,0.25)
