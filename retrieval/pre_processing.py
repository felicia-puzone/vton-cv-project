import cv2 as cv


# Contrast Stretching [0, 255] V1
def contrast_stretching(img, min_o=0, max_o=255):
    out = (img - img.min())*((max_o - min_o)/(img.max() - img.min())) + min_o
    return out.astype(img.dtype)


def resize_image(src, out, out_size=(384, 512), interpolation=''):
    if interpolation == '':
        src = cv.resize(src=src, dsize=out_size, dst=out)
    else:
        src = cv.resize(src=src, dsize=out_size, dst=out, interpolation=interpolation)

    return src


def perform_pre_processing(img):
    img = resize_image(img, img)

    img = cv.bilateralFilter(img, 20, 15, 15)

    img = contrast_stretching(img)

    return img

