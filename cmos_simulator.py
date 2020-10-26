"""
This module is designed to simulate CMOS intensity output given a 32-bit ground truth scene.
The CMOS sensor operates with a pre-defined full well capacity limit
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)

import cv2
import numpy as np

_path = "./input/simulator/"
fwc = 2**12  # full well capacity with a 12 bit sensor
gain = 1000
dr = -1  # dynamic range


def read_img():
    hdr_img = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
    assert hdr_img is not None
    return hdr_img


def expose(hdr_img):
    """
    simulate an exposure with a CMOS sensor.
    * apply a gain to the ground truth image
    * clop at full well capacity of the sensor
    * apply quantization / convert to integers
    :param hdr_img:
    :return: simulated CMOS image
    """
    ldr_img = hdr_img.copy()
    ldr_img = ldr_img * gain
    ldr_img[ldr_img >= fwc] = fwc
    ldr_img[ldr_img < 1.0] = 0
    ldr_img = ldr_img.astype(np.uint16)
    # adding poisson noise
    for p in np.nditer(ldr_img, op_flags=['readwrite']):
        p[...] = np.random.poisson(p)
    return ldr_img


def save_img(ldr_img):
    """
    outputs 16 bit png img
    :param ldr_img:
    :return:
    """
    ldr_img = (ldr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())
    ldr_img *= 2 ** 16
    ldr_img = ldr_img.astype(np.uint16)
    ldr_img[ldr_img <= 0] = 2 ** 16 - 1
    cv2.imwrite(_path + "ldr.png", ldr_img)


def main():
    img = read_img()
    img = expose(img)
    save_img(img)

if __name__ == "__main__":
    main()