"""
This module is designed to simulate CMOS intensity output given a 32-bit ground truth scene.
The CMOS sensor operates with a pre-defined full well capacity limit
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)

import cv2
import numpy as np

_path = "./input/simulator/"
fwc = 2**12  # full well capacity with a 12 bit sensor
T = 1  # exposure time in seconds
gain = 10000  # uniform gain applied to the analog signal
q = 1  # quantum efficiency index


def read_img():
    """
    read in a 32-bit hdr ground truth image. Pixels values are treated as photon flux
    :return: a matrix containing photon flux at each pixel location
    """
    hdr_img = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
    assert hdr_img is not None
    return hdr_img


def expose(flux):
    """
    simulate an exposure with a CMOS sensor.
    * simulate at every pixel for a given photon flux and exposure time
    * add poisson noise
    * clip at the full well capacity of the sensor
    * apply gain to the "analog" signal
    * convert to digital(uint16) signal
    :param hdr_img:
    :return: simulated CMOS image
    """
    img = flux.copy()
    img = img * T
    # adding poisson noise
    for p in np.nditer(img, op_flags=['readwrite']):
        p[...] = np.random.poisson(p)
    # clipping at the full well capacity of the sensor
    img[img >= fwc] = fwc
    img[img < 1.0] = 0
    # applying gain
    img = gain * img
    # apply quantization and ensure correct range for a 16-bit output
    img[img >= 2**16-1] = 2**16-1
    img = img.astype(np.uint16)
    return img


def save_img(ldr_img):
    """
    outputs 16 bit png img
    :param ldr_img:
    :return:
    """
    # ldr_img = (ldr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())
    # ldr_img *= 2 ** 16
    # ldr_img = ldr_img.astype(np.uint16)
    # ldr_img[ldr_img <= 0] = 2 ** 16 - 1
    cv2.imwrite(_path + "ldr.png", ldr_img)


def main():
    flux = read_img()
    img = expose(flux)
    save_img(img)

if __name__ == "__main__":
    main()