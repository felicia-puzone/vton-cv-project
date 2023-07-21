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
    cloth_dir = os.path.join(opt.root_dir,"cloth")
    dense_dir = os.path.join(opt.root_dir,"dense")
    image_mask_dir = os.path.join(opt.root_dir,"image-mask")
    image_dir = os.path.join(opt.root_dir,"images")
    keypoints_dir = os.path.join(opt.root_dir,"keypoints")
    label_maps_dir = os.path.join(opt.root_dir,"label_maps")
    skeletons_dir = os.path.join(opt.root_dir,"skeletons")

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

                root_file = file.split('_')[0]

                img_input = cv2.imread(os.path.join(cloth_mask_dir, file), cv2.IMREAD_GRAYSCALE)

                if is_valid_mask(img_input):

                    if write_count % train_test_ratio != 0:
                        f_train.write(root_file+'_0.jpg ' + root_file+'_1.jpg\n')
                        train_count += 1

                    if write_count % train_test_ratio == 0:
                        f_test.write(root_file+'_0.jpg ' + root_file+'_1.jpg\n')
                        test_count += 1

                    write_count += 1

                else:
                    print('Deleting cloth mask file:', root_file)
                    os.remove(os.path.join(cloth_mask_dir, file))

                    print('Deleting cloth file:', root_file)
                    os.remove(os.path.join(cloth_dir, root_file + '_1.jpg'))

                    print('Deleting dense file:', root_file)
                    os.remove(os.path.join(dense_dir, root_file + '_5.png'))

                    print('Deleting image-mask file:', root_file)
                    os.remove(os.path.join(image_mask_dir, root_file + '_0.png'))

                    print('Deleting image file:', root_file)
                    os.remove(os.path.join(image_dir, root_file + '_0.jpg'))

                    print('Deleting keypoints file:', root_file)
                    os.remove(os.path.join(keypoints_dir, root_file + '_2.json'))

                    print('Deleting label maps file:', root_file)
                    os.remove(os.path.join(label_maps_dir, root_file + '_4.png'))

                    print('Deleting skeletons file:', root_file)
                    os.remove(os.path.join(skeletons_dir, root_file + '_5.jpg'))

                    deleted_samples += 1

            print('Number of train images:', train_count)
            print('Number of test images:', test_count)
            print('Number of deleted objects:', deleted_samples)



        f_test.close()
    f_train.close()



