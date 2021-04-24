import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

PATH = "./dev/SPAD_dev/"
ftype_str = "hdr"
OUT_PATH = "./dev/out/figs_spad/"


def color_hist(img, fname):
    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        max_ = int(img.max())
        histr = cv2.calcHist([img], [channel], None, [max_], [0, max_])
        plt.plot(histr, color=col)
        # plt.ylim([0, 30000])
    plt.title("{} / max={} / min={}".format(fname, img.max(), img.min()))
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    plt.savefig("{}spad_hist_{}.png".format(OUT_PATH, fname))
    plt.close()


def disp_hist(img, fname):
    plt.hist(img.flatten(), bins=10000, log=True)
    plt.xscale('log')
    # plt.xlim((0, img.max()))
    # plt.ylim((0, 500))
    plt.title("{} / max={} / min={}".format(fname, img.max(), img.min()))
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    plt.savefig("{}ideal_hist_{}.png".format(OUT_PATH, fname))
    plt.close()

def main():
    _, _, files = next(os.walk(PATH))
    img_files = [x for x in files if ftype_str in x]
    fcount = len(img_files)
    assert(fcount != 0)
    print("{} image files of type {} detected".format(fcount, ftype_str))

    for idx in tqdm(range(fcount)):
        img = cv2.imread(PATH + str(idx) + "_spad.hdr", -1)
        # color_hist(img, idx)
        disp_hist(img, idx)



if __name__ == "__main__":
    main()

