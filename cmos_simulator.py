"""
This module is designed to simulate CMOS intensity output given a 32-bit ground truth scene.
The CMOS sensor operates with a pre-defined full well capacity limit
"""

# FIXME: does not handle pixels with values np.nan (dead pixels?)


import cv2
import numpy as np

class CMOSSimulator(object):

    def __init__(self, q=1, fwc=2**12, downsp_rate=1, path="./"):
        self.fwc = fwc
        self.q = q
        self.downsp_rate = downsp_rate
        self.path = path
        self.img = None

    def down_sample_flux(self, flux):
        r = self.downsp_rate
        flux = flux.copy()[::r, ::r, :]
        return flux

    def expose(self, flux, T, gain):
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
        img = flux.copy()
        img = img * T
        # adding poisson noise
        for p in np.nditer(img, op_flags=['readwrite']):
            p[...] = np.random.poisson(p)
        # clipping at the full well capacity of the sensor
        img[img >= self.fwc] = self.fwc
        img[img < 1.0] = 0
        # applying gain
        img = gain * img
        # apply quantization and ensure correct range for a 16-bit output
        img[img >= 2 ** 16 - 1] = 2 ** 16 - 1
        img = img.astype(np.uint16)
        self.img = img

    """IMAGE PROCESSING PIPELINE"""

    def process(self, id=""):
        self.save_img(id)

    def save_img(self, id):
        """
        outputs 16 bit png img
        :param img:
        :return: None
        """
        cv2.imwrite(self.path + id + "_cmos.png", self.img)