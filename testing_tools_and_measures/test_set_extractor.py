import os
import argparse
import shutil

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", default="C:\\Users\\ruteryan\\Desktop\\DressCode5.0_sala\\upper_body\\train\\images")
    parser.add_argument("--dest", default="dresscode_test")
    parser.add_argument("--data_list", default="C:\\Users\\ruteryan\\Desktop\\DressCodeFinal4.0_resized\\upper_body\\test_pairs_cv13_5.0_sala.txt")

    opt = parser.parse_args()
    return opt


def main():
    opt = get_opt()

    # load data list
    im_names = []

    with open(os.path.join(opt.dataroot, opt.data_list), 'r') as f:
        for line in f.readlines():
            im_name, c_name = line.strip().split()

            shutil.copy(os.path.join(opt.dataroot, im_name), opt.dest)



if __name__ == "__main__":
    main()