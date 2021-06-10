"""
This module generates a scaled version of the ground truth dataset without adding any noise
"""

import cv2
import numpy as np
from simulators.radiance_writer import radiance_writer


class IdealSimulator(object):

    def __init__(self, downsp_rate, path):
        self.downsp_rate = downsp_rate
        self.path = path
        self.img = None
        return

    def down_sample_flux(self, flux):
        r = self.downsp_rate
        h = int(flux.shape[0] / r)
        w = int(flux.shape[1] / r)
        dim = (w, h)
        resized = cv2.resize(flux, dim, interpolation=cv2.INTER_LINEAR)
        return resized

    def expose(self, flux, T):
        if self.downsp_rate != 1:
            flux = self.down_sample_flux(flux)
        img = flux.copy()
        img = img * T
        img[img <= 0] = 0
        self.img = img
        return

    def process(self, T, id=""):
        img = self.img / T
        self.save_hdr_img(img, id)

    def save_hdr_img(self, img, id):
        """
        :param img:
        :return: 32-bit hdr image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        radiance_writer(img, self.path + id + "_gt.hdr")