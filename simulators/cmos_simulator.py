"""
This module is designed to simulate CMOS intensity output given a 32-bit ground truth scene.
The CMOS sensor operates with a pre-defined full well capacity limit
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)


import cv2
import numpy as np
from radiance_writer import radiance_writer


class CMOSSimulator(object):

    def __init__(self, q, fwc, downsp_rate, path):
        self.fwc = fwc
        self.downsp_rate = downsp_rate
        self.q = np.array([q["b"], q["g"], q["r"]])
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
        """
            simulate an exposure with a CMOS sensor.
            * simulate at every pixel for a given photon flux and exposure time
            * add poisson noise
            * clip at the full well capacity of the sensor
            * apply gain to the "analog" signal
            * convert to digital(uint16) signal
            :param flux: photon flux at every pixel locations
            :return: simulated CMOS image
            """
        if self.downsp_rate != 1:
            flux = self.down_sample_flux(flux)
        img = flux.copy()
        img = img * self.q
        img = img * T
        # adding poisson noise
        img = np.random.poisson(img)
        # clipping at the full well capacity of the sensor
        img[img >= self.fwc] = self.fwc
        # img[img < 1.0] = 0   # FIXME: might be an issue
        self.img = img
    """IMAGE PROCESSING PIPELINE"""

    def process(self, T, id=""):
        img = self.img / self.q
        img /= T
        self.img = img
        # self.save_img(id)
        self.save_hdr_img(id)

    def save_img(self, id):
        """
        outputs 16 bit png img
        :param img:
        :return: None
        """
        img = self.img
        # apply quantization and ensure correct range for a 16-bit output
        img[img >= 2 ** 16 - 1] = 2 ** 16 - 1
        img = img.astype(np.uint16)
        cv2.imwrite(self.path + id + "_cmos.png", img)

    def save_hdr_img(self, id):
        """

        :param img:
        :return: 32-bit hdr image
        """
        img = self.img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        radiance_writer(img, self.path + id + "_cmos.hdr")