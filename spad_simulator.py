"""
This module is designed to simulate SPAD photon-counting output given a 32-bit ground truth scene.
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)

import cv2
import numpy as np
from radiance_writer import radiance_writer

_path = "./input/simulator/"
T = 10  # exposure time in seconds
gain = 1000  # uniform gain applied to the analog signal
q = 1  # quantum efficiency index
tau = 150e-9  # dead time in seconds


"""IMAGE ACQUISITION PHASE"""

def read_flux():
    """
    read in a 32-bit hdr ground truth image. Pixels values are treated as photon flux
    :return: a matrix containing photon flux at each pixel location
    """
    flux = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
    assert flux is not None
    return flux


def down_sample(img):
    img = img.copy()[::4, ::4, :]
    return img

def scale_flux(flux):
    flux *= 100000
    return flux


def expose(flux):
    """
    simulate an exposure with a SPAD sensor.
    :param flux: photon flux at every pixel location
    :return: simulated SPAD image using photon counts
    """
    img = flux.copy()
    # adding poisson noise
    for p in np.nditer(img, op_flags=['readwrite']):
        phi = p  # photon flux
        num = q * phi * T  # numerator
        den = 1 + q * phi * tau  # denominator
        mean = num / den  # expectation of photon counts
        var = num / den**3  # variance of photon counts
        p[...] = np.random.normal(mean, var**.5)
    # apply quantization and ensure correct range of [0, T/tau]
    img = img.astype(np.uint32)
    img[img <= 0] = 0
    ub = T / tau  # upper bound, asymptotic saturation of SPAD
    img[img >= ub] = ub
    return img


def save_photon_counts(photon_counts):
    """
    outputs 32 bit
    :param ldr_img:
    :return:
    """
    fname = "photon_count.npy"
    np.save(_path + fname, photon_counts)
    img = np.load(_path + fname)
    return img

"""IMAGE PROCESSING PIPELINE"""


def linearize(img):
    print("linearizing image...")
    for N in np.nditer(img, op_flags=['readwrite']):
        N[...] = N / (T - N * tau)
    return img


def save_img(img):
    # img *= gain
    img = (img - img.min()) / (img.max() - img.min())
    img *= 2 ** 16
    img = img.astype(np.uint16)
    img[img >= 2 ** 16 - 1] = 2 ** 16 - 1
    cv2.imwrite(_path + "yes.png", img)

def main():
    flux = read_flux()
    flux = scale_flux(flux)
    flux = down_sample(flux)
    img = expose(flux)
    save_photon_counts(img)
    # img = linearize(img)
    save_img(img)

if __name__ == "__main__":
    main()