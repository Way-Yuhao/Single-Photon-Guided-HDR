"""
This module runs luminance fusion between a CMOS and SPAD image.
Requirements:
    * the 2 input images have the same dimensions
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from simulators.radiance_writer import radiance_writer

_ldr_path = "./simulated_inputs/CMOS/5_cmos.png"
_hdr_path = "./simulated_inputs/SPAD_HDR_SR/5_spad_bilinear.hdr"
_out_img_path = "./playground/"

tau = .9  # threshold
gamma = 4  # actually is 1/gamma
gain = 1.5


def read_img():
    ldr_img = cv2.imread(_ldr_path, -1)
    hdr_img = cv2.imread(_hdr_path, -1)
    assert (len(ldr_img) == len(hdr_img) and len(ldr_img[0]) == len(hdr_img[0]))
    # tonemapDrago = cv2.createTonemapDrago(gamma, 1)
    # hdr_img = tonemapDrago.process(hdr_img)
    return ldr_img, hdr_img


def rescale(ldr_img, hdr_img):
    """
    rescale both image to [0, 1]
    """
    hdr_img = (hdr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())  # rescale to [0, 1]
    ldr_img = (ldr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())  # rescale to [0, 1]
    # print(np.nanmax(hdr_img))
    # print(np.nanmin(hdr_img))
    return ldr_img, hdr_img


def calc_weight(pixel_val):
    wi = (0.5 - max(abs(pixel_val - .5), tau - .5)) / (1 - tau)
    return wi


def fusion(ldr_img, hdr_img):
    fused_img = np.zeros((len(ldr_img), len(ldr_img[0]), 3))
    for x in range(len(ldr_img)):
        for y in range(len(ldr_img[0])):
            w_ldr = calc_weight(np.sum(ldr_img[x][y])/3)
            # w_hdr = calc_weight(np.sum(hdr_img[x][y])/3)
            w_hdr = 1 - w_ldr
            for c in range(3):
                fused_img[x][y][c] = (w_ldr * ldr_img[x][y][c] + w_hdr * hdr_img[x][y][c] * gain) / (w_ldr + w_hdr)
    return fused_img


def naive_fusion(ldr_img, hdr_img):
    fused_img = np.zeros((len(ldr_img), len(ldr_img[0]), 3))
    for x in range(len(ldr_img)):
        for y in range(len(ldr_img[0])):
            if max(ldr_img[x][y]) < .9:
                fused_img[x][y] = ldr_img[x][y]
            else:
                fused_img[x][y] = hdr_img[x][y]
    return fused_img


def disp_hist(ldr, hdr):
    # plt.imshow(hdr)
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 6.5)
    plt.subplot(1, 2, 1)
    plt.hist(ldr.flatten(), bins=100)
    plt.title("Histogram of 8-bit LDR Image")
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")

    plt.subplot(1, 2, 2)
    plt.hist(hdr.flatten(), bins=10000)
    plt.xlim((0, 2**25))
    plt.ylim((0, 1e8))
    plt.title("Histogram of 32-bit HDR Image")
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    plt.show()


def color_hist(ldr, hdr):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 6.5)
    color = ('b', 'g', 'r')

    # ldr
    plt.subplot(1, 2, 1)
    for channel, col in enumerate(color):
        histr = cv2.calcHist([ldr], [channel], None, [2**16], [0, 2**16])
        plt.plot(histr, color=col)
        # plt.xlim([0, 2**14])
        plt.ylim([0, 30000])
    plt.title("Histogram of simulated CMOS data")
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    # plt.show()

    # hdr
    plt.subplot(1, 2, 2)
    for channel, col in enumerate(color):
        histr = cv2.calcHist([hdr], [channel], None, [2**16], [0, 2**18])
        plt.plot(histr, color=col)
        # plt.xlim([0, 2**16])
        plt.ylim([0, 30000])
    plt.title("Histogram of simulated SPAD data")
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    plt.show()

def show_weight_map(ldr_img):
    w_map = np.zeros((len(ldr_img), len(ldr_img[0]), 3))
    for x in range(len(ldr_img)):
        for y in range(len(ldr_img[0])):
            w_ldr = calc_weight(np.sum(ldr_img[x][y])/3)
            w_hdr = 1 - w_ldr
            w_map[x][y][0] = w_hdr
    plt.imshow(w_map)
    plt.title("Weight Map of the HDR image")
    plt.show()


def save_img(img):
    # 8 bit workflow
    cv2.imwrite(_out_img_path + "8bit.png", img * 255)

    # 16 bit workflow
    # img_16 = (img - img.min()) / (img.max() - img.min())
    img_16 = img * 2**16
    img_16 = img_16.astype(np.uint16)
    img_16[img_16 <= 0] = 2**16 - 1
    img_16[img_16 >= 2 ** 16] = 2 ** 16 - 1
    cv2.imwrite(_out_img_path + "16.png", img_16)


    img_32 = img.astype(np.float32)
    img_32 = cv2.cvtColor(img_32, cv2.COLOR_BGR2RGB)
    radiance_writer(img_32, _out_img_path + "_32.hdr")


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
    # disp_hist(ldr_img, hdr_img)
    color_hist(ldr_img, hdr_img)
    ldr_img, hdr_img = rescale(ldr_img, hdr_img)
    # plot_weight_func()
    # fused_img = fusion(ldr_img, hdr_img)
    # fused_img = naive_fustion(ldr_img, hdr_img)
    # show_weight_map(ldr_img)
    # save_img(fused_img)

