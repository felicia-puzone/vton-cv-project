import torch
import cv2 as cv
import torch.nn as nn
from matplotlib import pyplot as plt
class GrabCut(nn.Module):
    def __init__(self):
        super(GrabCut, self).__init__()
    def forward(self,img):
        print(img)
        print(type(img))
        blur = cv.GaussianBlur(img, (11, 11), 0)
        #ret, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
        #ret, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
        #ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
        #ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
        #ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
        ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = cv.bitwise_not(thresh)
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv.drawContours(thresh, [c], 0, (255, 255, 255), -1)
        return thresh
if __name__=="__main__":

    file = "020715"
    dir=f"../../../dataset_sample"
    img=cv.imread(f"{dir}/dresses/images/{file}_0.jpg",0)
    gc=GrabCut()
    out=gc(img)
    plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(out)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
