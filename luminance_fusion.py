"""

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

_ldr_path = "./input/starter/ldr.png"
_hdr_path = "./input/starter/high_res.hdr"
_out_img_path = "./input/starter/fusion.png"

tau = .8  # threshold


def read_img():
    ldr_img = cv2.imread(_ldr_path, -1)
    hdr_img = cv2.imread(_hdr_path, -1)
    assert (len(ldr_img) == len(hdr_img) and len(ldr_img[0]) == len(hdr_img[0]))
    return ldr_img, hdr_img

"""
Linearizes both image to [0, 1]
"""
def linearize(ldr_img, hdr_img):
    ldr_img = (ldr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())
    hdr_img = (hdr_img - hdr_img.min()) / (hdr_img.max() - hdr_img.min())
    return ldr_img, hdr_img


def calc_weight(pixel_val):
    wi = (0.5 - max(abs(pixel_val - .5), tau - .5)) / (1 - tau)
    return wi


def fusion(ldr_img, hdr_img):
    fused_img = np.zeros((len(ldr_img), len(ldr_img[0])))
    for x in range(len(ldr_img)):
        for y in range(len(ldr_img[0])):
            w_ldr = calc_weight(ldr_img[x][y])
            w_hdr = calc_weight(hdr_img[x][y])
            # w_hdr = 1 - w_ldr
            fused_img[x][y] = (w_ldr * ldr_img[x][y] + w_hdr * hdr_img[x][y]) / (w_ldr + w_hdr)
    return fused_img

def save_img(img):
    img = (img - img.min()) / (img.max() - img.min())
    img *= 2**16
    img = ldr_img.astype(np.uint16)
    img[img <= 0] = 2**16 - 1
    cv2.imwrite(_out_img_path, img)


def plot_weight_func():
    X = np.linspace(0, 1, 100)
    Y = []
    for x in X:
        Y += [calc_weight(x)]
    plt.plot(X, Y)
    plt.xlabel('pixel values of the LDR image')
    plt.ylabel('weight value, wi')
    title_txt = "Weighting Function when tau = {}".format(tau)
    plt.title(title_txt)
    plt.show()


if __name__ == "__main__":
    ldr_img, hdr_img = read_img()
    ldr_img, hdr_img = linearize(ldr_img, hdr_img)
    # plot_weight_func()
    fused_img = fusion(ldr_img, hdr_img)
    save_img(fused_img)

