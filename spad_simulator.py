"""
This module is designed to simulate SPAD photon-counting output given a 32-bit ground truth scene.
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)

import cv2
import numpy as np
from radiance_writer import radiance_writer
from PIL import Image, ImageEnhance


class SPADSimulator(object):

    def __init__(self, q=1, tau=150e-9, downsp_rate=4, path="./"):
        self.tau = tau
        self.q = q
        self.downsp_rate = downsp_rate
        self.path = path
        self.img = None

    def down_sample_flux(self, flux):
        r = self.downsp_rate
        flux = flux.copy()[::r, ::r, :]
        return flux

    def expose(self, flux, T):
        if self.downsp_rate != 1:
            flux = self.down_sample_flux(flux)
        img = flux.copy()
        # adding poisson noise
        for p in np.nditer(img, op_flags=['readwrite']):
            phi = p  # photon flux
            num = self.q * phi * T  # numerator
            den = 1 + self.q * phi * self.tau  # denominator
            mean = num / den  # expectation of photon counts
            var = num / den ** 3  # variance of photon counts
            p[...] = np.random.normal(mean, var ** .5)
        # apply quantization and ensure correct range of [0, T/tau]
        img = np.rint(img)
        img[img <= 0] = 0
        ub = T / self.tau  # upper bound, asymptotic saturation of SPAD
        img[img >= ub] = ub
        self.img = img

    def save_photon_counts(self):
        """
        outputs 32 bit photon counts
        :param ldr_img:
        :return:
        """
        fname = "photon_count" + id + ".npy"
        np.save(self.path + fname, self.img)

    """IMAGE PROCESSING PIPELINE"""

    def process(self, T, gain, id=""):
        img = self.img.copy()  # processed image
        img = self.linearize(img, T)
        self.save_hdr_img(img, id)
        self.save_img(img, gain, id)

    def linearize(self, img, T):
        for N in np.nditer(img, op_flags=['readwrite']):
            N[...] = N / (T - N * self.tau)
        return img

    def save_hdr_img(self, img, id):
        """

        :param img:
        :return: 32-bit hdr image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        radiance_writer(img, self.path + id + "_spad.hdr")

    def save_img(self, img, gain, id):
        """
        :param img:
        :return: 16-bit tone-mapped png
        """
        tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
        img = tonemapDrago.process(img)
        img = (img - img.min()) / (img.max() - img.min())
        img *= 2 ** 16
        img = img.astype(np.uint16)
        img[img >= 2 ** 16 - 1] = 2 ** 16 - 1
        cv2.imwrite(self.path + id + "_spad_16bit.png", img)