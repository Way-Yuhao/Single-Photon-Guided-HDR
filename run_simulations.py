"""
This module runs spad and cmos simulators
"""

import cv2
import os
from spad_simulator import SPADSimulator
from cmos_simulator import CMOSSimulator
"""global parameters"""
fpath = "./input/53_HDRI/"
out_path = "./input/out/"

"""SPAD parameters"""
SPAD_Sim = None
SPAD_T = .01  # exposure time in seconds
SPAD_gain = 10  # uniform gain applied to the analog signal
SPAD_q = 1  # quantum efficiency index
SPAD_tau = 150e-9  # dead time in seconds
SPAD_down_sample_rate = 4

"""CMOS parameters"""
CMOS_Sim = None
CMOS_fwc = 2**12  # full well capacity with a 12 bit sensor
CMOS_T = .01  # exposure time in seconds
CMOS_gain = 100  # uniform gain applied to the analog signal
CMOS_q = 1  # quantum efficiency index


def read_flux(fname):
    """
    read in a 32-bit hdr ground truth image. Pixels values are treated as photon flux
    :return: a matrix containing photon flux at each pixel location
    """
    flux = cv2.imread(fname, -1)
    assert flux is not None
    return flux


def scale_flux(flux):
    """
    scales the flux matrix
    :return:
    """
    flux *= 100000
    return flux


def init_simulators():
    global SPAD_Sim, CMOS_Sim
    SPAD_Sim = SPADSimulator(SPAD_q, SPAD_tau, SPAD_down_sample_rate, path=out_path)
    CMOS_Sim = CMOSSimulator(q=CMOS_q, fwc=CMOS_fwc, downsp_rate=1, path=out_path)

def run_sumulations(flux, id):
    global SPAD_Sim, CMOS_Sim
    SPAD_Sim.expose(flux, SPAD_T)
    SPAD_Sim.process(SPAD_T, SPAD_gain, id)

    CMOS_Sim.expose(flux, CMOS_T, CMOS_gain)
    CMOS_Sim.process(id)


def main():
    id = 0
    for filename in os.listdir(fpath):
        flux = read_flux(os.path.join(fpath, filename))
        flux = scale_flux(flux)
        init_simulators()
        run_sumulations(flux, str(id))
        id += 1
        break

    # flux = read_flux()
    # flux = scale_flux(flux)
    # init_simulators()
    # run_sumulations(flux)


if __name__ == "__main__":
    main()