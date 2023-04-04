# coded following: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('020717_0.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv.Canny(img, 100, 120)

plt.subplot(121), plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


#im2, contours = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#plt.imshow(im2)
#plt.show()
#plt.imshow(contours)
#plt.show()