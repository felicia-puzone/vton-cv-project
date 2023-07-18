import numpy as np
import argparse
import os
import cv2


def is_valid_mask(mask):
    mask_white_count = (mask == 255).sum()
    mask_black_count = (mask == 0).sum()

    wb_ratio = mask_white_count/mask_black_count

    if wb_ratio < 0.3:
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=None)

    opt = parser.parse_args()

    if opt.root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.root_dir) or not os.path.isdir(opt.root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)

    cloth_mask_dir = os.path.join(opt.root_dir,"cloth-mask")

    with open('train_pairs_new.txt', 'w') as f:
        f.write('Create a new text file!')

    train_count = 0
    test_count = 0
    deleted_samples = 0

    train_test_ratio = 10

    # train-test writing loop
    with open('train_pairs_new.txt', 'w') as f_train:
        with open('test_pairs_new.txt', 'w') as f_test:

            write_count = 0

            for file in os.listdir(cloth_mask_dir):

                img_input = cv2.imread(os.path.join(cloth_mask_dir, file), cv2.IMREAD_GRAYSCALE)

                if is_valid_mask(img_input):

                    if write_count % train_test_ratio != 0:
                        f_train.write(file.split('_')[0]+'_0.jpg ' + file.split('_')[0]+'_1.jpg\n')
                        train_count += 1

                    if write_count % train_test_ratio == 0:
                        f_test.write(file.split('_')[0]+'_0.jpg ' + file.split('_')[0]+'_1.jpg\n')
                        test_count += 1

                    write_count += 1

                else:
                    os.remove(os.path.join(cloth_mask_dir, file))
                    deleted_samples += 1

            print('Number of train images:', train_count)
            print('Number of test images:', test_count)
            print('Number of deleted objects:', deleted_samples)

        f_test.close()
    f_train.close()



