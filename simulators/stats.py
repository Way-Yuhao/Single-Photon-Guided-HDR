import os
import os.path as p
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm

path = "../test/"
out_path = "../test/CMOS_8bit_PNG"


def measure_cmos():
    cmos_path = p.join(path, "CMOS")
    cmos_list = natsorted(os.listdir(cmos_path))
    print(len(cmos_list))

    length = len(cmos_list)

    means = np.zeros(length)
    mins = np.zeros(length)
    maxs = np.zeros(length)
    for i in tqdm(range(length)):
        fname = "{}_cmos.hdr".format(i)
        cmos = cv2.imread(p.join(cmos_path, fname), -1)
        means[i] = cmos.mean()
        mins[i] = cmos.min()
        maxs[i] = cmos.max()

    print("mean = {}".format(means.mean()))
    print("mean max = {}".format(maxs.mean()))
    print("max = {}".format(maxs.max()))
    print("mean min = {}".format(mins.mean()))


def create_pngs():
    cmos_path = p.join(path, "CMOS")
    cmos_list = natsorted(os.listdir(cmos_path))
    print(len(cmos_list))
    ratio = 3342336.0 / 2**16

    length = len(cmos_list)

    for i in tqdm(range(length)):
        fname = "{}_cmos.hdr".format(i)
        cmos = cv2.imread(p.join(cmos_path, fname), -1)
        cmos = cmos / ratio
        cmos[cmos == 2**16] = 2**16 - 1
        cmos = cmos.astype('uint16')
        out_fname = p.join(out_path, "{}_cmos.png".format(i))
        cv2.imwrite(out_fname, cmos)


def create_8bit_pngs():
    cmos_path = p.join(path, "CMOS")
    cmos_list = natsorted(os.listdir(cmos_path))
    print(len(cmos_list))
    ratio = 3342336.0 / 2**8

    length = len(cmos_list)

    for i in tqdm(range(length)):
        fname = "{}_cmos.hdr".format(i)
        cmos = cv2.imread(p.join(cmos_path, fname), -1)
        cmos = cmos / ratio
        cmos[cmos >= 2 ** 8] = 2 ** 8 - 1
        cmos = cmos.astype('uint8')
        out_fname = p.join(out_path, "{}_cmos.png".format(i))
        cv2.imwrite(out_fname, cmos)

def main():
    create_8bit_pngs()


if __name__ == "__main__":
    main()
