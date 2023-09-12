from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import argparse
import os


def options():
    parser = argparse.ArgumentParser(description="Read image metadata")
    parser.add_argument("-o", "--first", help="Input image folder.", default= "C:\\Users\\ruteryan\\Desktop\\Testing_folders\\TOM_CP_PLUS_50.000_testset5.0\\try-on")
    parser.add_argument("-c", "--second", help="Input image folder.", default= "C:\\Users\\ruteryan\\Desktop\\Testing_folders\\dresscode_test_5.0")
    args = parser.parse_args()
    return args

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error


def compare(imageA, imageB):
    # Calculate the MSE and SSIM
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # Return the SSIM. The higher the value, the more "similar" the two images are.
    return s


def main():
    # Get options
    args = options()

    SSIM_list = []
    MSE_list = []

    # Import images
    for f in os.listdir(args.first):
        image1 = cv2.imread(args.first + "\\" + f, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(args.second + "\\" + f, cv2.IMREAD_GRAYSCALE)


        # Check for same size and ratio and report accordingly
        ho, wo  = image1.shape
        hc, wc = image2.shape
        ratio_orig = ho / wo
        ratio_comp = hc / wc
        dim = (wc, hc)

        if round(ratio_orig, 2) != round(ratio_comp, 2):
            print("\nImages not of the same dimension. Check input.")
            exit()

        # Resize first image if the second image is smaller
        elif ho > hc and wo > wc:
            print("\nResizing original image for analysis...")
            gray1 = cv2.resize(image1, dim)

        elif ho < hc and wo < wc:
            print("\nCompressed image has a larger dimension than the original. Check input.")
            exit()

        if round(ratio_orig, 2) == round(ratio_comp, 2):
            mse_value = mse(image1, image2)
            ssim_value = compare(image1, image2)

            SSIM_list.append(ssim_value)
            MSE_list.append(mse_value)

            print("MSE:", mse_value)
            print("SSIM:", ssim_value)

    mean_SSIM = sum(SSIM_list) / len(SSIM_list)
    mean_MSE = sum(MSE_list) / len(MSE_list)

    print("MEAN SSIM OVER COLLECTION:", mean_SSIM)
    print("MEAN MSE OVER COLLECTION:", mean_MSE)

if __name__ == '__main__':
    main()