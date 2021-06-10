"""
This module runs spad and cmos simulators
"""

import cv2
import os
from matplotlib import pyplot as plt
from simulators.spad_simulator import SPADSimulator
from simulators.cmos_simulator import CMOSSimulator
from simulators.ideal_simulator import IdealSimulator
from tqdm import tqdm

"""global parameters"""
collection_path = "../input/collection/"
out_path = "../simulated_outputs/"
plt_path = "../simulated_outputs/plt/"

# artificial_path = "../simulated_outputs/artificial/input"

"""SPAD parameters"""
SPAD_Sim = None
SPAD_on = True             # toggle on to enable SPAD simulator
SPAD_mono = False          # if the sensor is monochromatic
SPAD_T = .01               # exposure time in seconds
# SPAD_T = .01               # exposure time in seconds
# SPAD_gain = 10             # uniform gain applied to the analog signal
SPAD_tau = 150e-9          # dead time in seconds
SPAD_down_sample_rate = 4  # spatial down sampling rate of the sensor
SPAD_qe = .4               # quantum efficiency index

"""CMOS parameters"""
CMOS_Sim = None
CMOS_on = True              # toggle on to enable CMOS simulator
CMOS_mono = False           # if the sensor is monochromatic
CMOS_fwc = 33400            # full well capacity with a 15 bit sensor
CMOS_T = .01                # exposure time in seconds
# CMOS_T = .000001                # exposure time in seconds
# CMOS_gain = 1             # uniform gain applied to the analog signal
CMOS_down_sample_rate = 1   # spatial down sampling rate of the sensor
CMOS_qe = {                 # quantum efficiency index for each color channel
    'r': .40,
    'g': .75,
    'b': .77
}

"""ideal sensor parameters"""
ideal_Sim = None
ideal_on = True
idea_T = CMOS_T
# ideal_gain = CMOS_gain
ideal_down_sample_rate = CMOS_down_sample_rate


def resave_gt(fname, id):
    """
    resaves ground truth files with an ordered naming scheme
    :param fname:
    :param id:
    :return:
    """
    os.rename(fname, '../input/gt/{}.hdr'.format(id))


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
    scales the flux matrix by a constant
    :return: scaled ground truth matrix
    """

    # flux *= 1e5  # HDRI
    flux *= 1e7  # Laval Indoor
    # flux *= 5e4  # HDR_MATLAB_3x3
    return flux


def init_simulators():
    global SPAD_Sim, CMOS_Sim, ideal_Sim
    if SPAD_on:
        SPAD_Sim = \
            SPADSimulator(q=SPAD_qe, tau=SPAD_tau, downsp_rate=SPAD_down_sample_rate, isMono=SPAD_mono, path=out_path+"SPAD/")
    if CMOS_on:
        CMOS_Sim = CMOSSimulator(q=CMOS_qe, fwc=CMOS_fwc, downsp_rate=CMOS_down_sample_rate, path=out_path+"CMOS/")
    if ideal_on:
        ideal_Sim = IdealSimulator(downsp_rate=ideal_down_sample_rate, path=out_path+"ideal/")


def run_simulations(flux, id):
    global SPAD_Sim, CMOS_Sim, ideal_Sim
    if SPAD_on:
        SPAD_Sim.expose(flux, SPAD_T)
        SPAD_Sim.process(SPAD_T, id)
    if CMOS_on:
        CMOS_Sim.expose(flux, CMOS_T)
        CMOS_Sim.process(CMOS_T, id)
    if ideal_on:
        ideal_Sim.expose(flux, idea_T)
        ideal_Sim.process(idea_T, id)


def save_hist(flux, id):
    plt.hist(flux.flatten(), bins=100, log=True)
    plt.title("Photon flux values for scene {} after rescaling".format(id))
    plt.xlabel("photon flux values")
    plt.ylabel("pixel counts")
    plt.savefig(plt_path + 'plt_{}.png'.format(id))
    plt.clf()


def run(fpath):
    path, dirs, files = next(os.walk(fpath))
    file_count = len([x for x in files if "hdr" in x or "exr" in x])
    print("processing {} hdr files".format(file_count))
    i = 0
    for filename in tqdm(os.listdir(fpath)):
        if not filename.endswith(".hdr") and not filename.endswith(".exr"):
            continue
        flux = read_flux(os.path.join(fpath, filename))
        flux = scale_flux(flux)
        init_simulators()
        run_simulations(flux, str(i))
        save_hist(flux, i)
        i += 1


def init():
    # out_path = "../simulated_outputs/"
    if os.path.exists(out_path + "CMOS") or os.path.exists(out_path + "ideal") or os.path.exists(out_path + "SPAD") or \
            os.path.exists(out_path + "plt"):
        raise FileExistsError("ERROR: found target directories inside {}. Please remove.".format(out_path))

    os.mkdir(out_path + "CMOS")
    os.mkdir(out_path + "SPAD")
    os.mkdir(out_path + "ideal")
    os.mkdir(out_path + "plt")

    return


def main():
    init()
    # run(collection_path + "100samplesDataset")
    # run(collection_path + "HDRI_4k")
    run(collection_path + "HDR_MATLAB_3x3")
    # run(artificial_path)

if __name__ == "__main__":
    main()