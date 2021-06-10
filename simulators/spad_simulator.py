"""
This module is designed to simulate SPAD photon-counting output given a 32-bit ground truth scene.
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)

import cv2
import numpy as np
from simulators.radiance_writer import radiance_writer


class SPADSimulator(object):

    def __init__(self, q, tau, downsp_rate, isMono, path):
        self.tau = tau
        self.downsp_rate = downsp_rate
        self.isMono = isMono
        self.q = q
        self.path = path
        self.img = None

    def down_sample_flux(self, flux):
        r = self.downsp_rate
        h = int(flux.shape[0] / r)
        w = int(flux.shape[1] / r)
        dim = (w, h)
        resized = cv2.resize(flux, dim, interpolation=cv2.INTER_AREA)
        return resized

    def expose(self, flux, T):
        if self.downsp_rate != 1:
            flux = self.down_sample_flux(flux)
        img = flux.copy()
        if self.isMono:  # monochrome
            b, g, r = cv2.split(flux)
            # img = 0.2126 * r + 0.7152 * g + 0.0722 * b  # convert to monochrome
            img = g
        # adding photon noise
        # for p in np.nditer(img, op_flags=['readwrite']):
        #     phi = p  # photon flux
        #     num = self.q * phi * T  # numerator
        #     den = 1 + self.q * phi * self.tau  # denominator
        #     mean = num / den  # expectation of photon counts
        #     var = num / den ** 3  # variance of photon counts
        #     p[...] = np.random.normal(mean, var ** .5)

        phi = img  # photon flux
        num = self.q * phi * T  # numerator
        den = 1 + self.q * phi * self.tau  # denominator
        mean = num / den  # expectation of photon counts
        var = num / den ** 3  # variance of photon counts
        img = np.random.normal(mean, var**.5)

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

    def process(self, T, id=""):
        img = self.img.copy()  # processed image
        img = self.linearize(img, T)
        img /= self.q  # factor in qe
        if self.isMono:
            img = np.dstack((img, img, img))
        img = img.astype(np.float32)

        self.save_hdr_img(img, id)
        # self.save_img(img, gain, id)

    def linearize(self, img, T):
        # for N in np.nditer(img, op_flags=['readwrite']):
        #     N[...] = N / (T - N * self.tau)

        img = img / (T - img * self.tau)
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
        img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
        img *= 2 ** 16
        img = img.astype(np.uint16)
        img[img >= 2 ** 16 - 1] = 2 ** 16 - 1
        cv2.imwrite(self.path + id + "_spad_16bit.png", img)