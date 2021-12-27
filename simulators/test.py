import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import os.path as p
from radiance_writer import radiance_writer

cmos_saturation = 33400 / .01


def normalize(img):
    img = img / cmos_saturation * 255
    return img


def main():
        img_path = "../test/provided/0001.png"
        # out_path = "../test/dhdr_14.hdr"
        img = cv2.imread(img_path, -1)

        print(1)
        # img = np.sqrt(img)
        # radiance_writer(img, out_path)


if __name__ == "__main__":
    main()