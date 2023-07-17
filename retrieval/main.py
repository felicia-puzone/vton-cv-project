import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging as lgg

import builder
import pre_processing as pre
import matching as mtc


logger = lgg.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
lgg.basicConfig(format=FORMAT)
logger.setLevel(lgg.DEBUG)


if __name__ == '__main__':
    img_full_path = './prova.jpeg'

    # builder.build_repository('D:\IngMagistrale\Computer_vision\project\dataset_example\images\\')
    # builder.build_images()

    print("Loading query image...")
    query_img = cv.imread(img_full_path, cv.IMREAD_GRAYSCALE)
    query_img = pre.perform_pre_processing(query_img)

    best_n_img = mtc.match_n_best(query_img, ratio_match=0.4)
    for img_name, img_score in best_n_img:
        logger.info(f"Match name: {img_name},\t match score: {img_score}")
    mtc.plot_match_n_best(query_img, best_n_img)