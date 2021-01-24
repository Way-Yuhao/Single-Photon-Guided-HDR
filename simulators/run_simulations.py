"""
This module runs spad and cmos simulators
"""

import cv2
import os
from matplotlib import pyplot as plt
from simulators.spad_simulator import SPADSimulator
from simulators.cmos_simulator import CMOSSimulator
import progressbar

"""global parameters"""
fpath = "../input/53_HDRI/"
out_path = "../simulated_inputs/CMOS_mid/"
plt_path = "../input/plt/"

"""SPAD parameters"""
SPAD_Sim = None
SPAD_T = .01               # exposure time in seconds
SPAD_gain = 10             # uniform gain applied to the analog signal
SPAD_qe = .4               # quantum efficiency index
SPAD_tau = 150e-9          # dead time in seconds
SPAD_down_sample_rate = 4  # spatial down sampling rate of the sensor
SPAD_mono = True           # if the sensor is monochromatic

"""CMOS parameters"""
CMOS_Sim = None
CMOS_fwc = 2**12            # full well capacity with a 12 bit sensor
CMOS_T = .01                # exposure time in seconds
CMOS_gain = 100             # uniform gain applied to the analog signal
CMOS_qe = {                 # quantum efficiency index for each color channel
    'r' : .40,
    'g' : .75,
    'b' : .77
}

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
    flux *= 100000
    return flux


def check_cam_params():
    """
    checks if the camera parameters are valid
    :return: None
    """




def init_simulators():
    global SPAD_Sim, CMOS_Sim
    # SPAD_Sim = SPADSimulator(SPAD_q, SPAD_tau, SPAD_down_sample_rate, path=out_path)
    CMOS_Sim = CMOSSimulator(q=CMOS_qe, fwc=CMOS_fwc, downsp_rate=1, path=out_path)


def run_sumulations(flux, id):
    global SPAD_Sim, CMOS_Sim
    # SPAD_Sim.expose(flux, SPAD_T)
    # SPAD_Sim.process(SPAD_T, SPAD_gain, id)

    CMOS_Sim.expose(flux, CMOS_T, CMOS_gain)
    CMOS_Sim.process(id)


def save_hist(flux, id):
    plt.hist(flux.flatten(), bins=100, log=True)
    plt.title("Photon flux values for scene {} after rescaling".format(id))
    plt.xlabel("photon flux values")
    plt.ylabel("pixel counts")
    plt.savefig(plt_path + 'plt_{}.png'.format(id))
    plt.clf()


def main():
    path, dirs, files = next(os.walk(fpath))
    file_count = len([x for x in files if "hdr" in x])
    print("processing {} hdr files".format(file_count))
    id = 0
    with progressbar.ProgressBar(max_value=file_count) as bar:
        bar.update(id)
        for filename in os.listdir(fpath):
            if not filename.endswith(".hdr"):
                continue
            flux = read_flux(os.path.join(fpath, filename))
            flux = scale_flux(flux)
            init_simulators()
            run_sumulations(flux, str(id))
            save_hist(flux, id)
            # resave_gt(os.path.join(fpath, filename), id)
            bar.update(id)
            id += 1

if __name__ == "__main__":
    main()