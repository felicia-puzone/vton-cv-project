import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--train-test-ratio", default=10)

    args = parser.parse_args()

    if args.root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)


    train_count = 0
    test_count = 0

    train_test_ratio = int(args.train_test_ratio)

    cloth_mask_dir = os.path.join(args.root_dir,"cloth-mask")

    # train-test writing loop
    with open(os.path.join(args.root_dir,'train_pairs_cv13.txt'), 'w') as f_train:
        with open(os.path.join(args.root_dir,'test_pairs_cv13.txt'), 'w') as f_test:
            write_count = 0

            for file in os.listdir(cloth_mask_dir):
                root_file = file.split('_')[0]

                if write_count % train_test_ratio != 0:
                    f_train.write(root_file + '_0.jpg ' + root_file + '_1.jpg\n')
                    train_count += 1

                if write_count % train_test_ratio == 0:
                    f_test.write(root_file + '_0.jpg ' + root_file + '_1.jpg\n')
                    test_count += 1

                write_count += 1

            print('Number of train images:', train_count)
            print('Number of test images:', test_count)

        f_test.close()
    f_train.close()
