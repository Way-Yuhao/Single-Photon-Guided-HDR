import os
import os.path as p
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import scipy.io
from matplotlib import pyplot as plt
from radiance_writer import radiance_writer

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

###############################################################################################


def read_mat_fire():
    path = "../read_data/fire/count_img_linearized_warped_simple.mat"
    mat = scipy.io.loadmat(path)
    spad = mat["img_data"]
    spad = np.dstack((spad, spad, spad))
    spad /= 90
    # spad /= 60


    cmos = cv2.imread("../read_data/fire/cmos_Tunnel2-2-.5ms.png", -1)
    cmos = np.dstack((cmos, cmos, cmos))
    cmos = cmos / (.0005 * .7)


    # plt.imshow(spad / cmos.max())
    # plt.show()
    #
    # plt.imshow(cmos / cmos.max())
    # plt.show()

    h, w, _ = cmos.shape
    cmos_ss = cv2.resize(cmos, (256, 256), interpolation=cv2.INTER_LINEAR)
    spad_ss = cv2.resize(spad, (64, 64), interpolation=cv2.INTER_LINEAR)
    print("cmos saturation = {}".format(cmos_ss.max()))

    plt.imshow(cmos_ss / cmos_ss.max())
    plt.show()
    plt.imshow(spad_ss / spad_ss.max())
    plt.show()
    radiance_writer(cmos_ss, "../read_data/out/cmos1_fire_256x256.hdr")
    radiance_writer(spad_ss, "../read_data/out/spad1_fire_64x64.hdr")


def read_small_mid_crop():
    path = "../read_data/smallmidcrop/count_img_linearized_warped.mat"
    mat = scipy.io.loadmat(path)
    spad = mat["img_data"]
    spad = np.dstack((spad, spad, spad))
    spad /= 0.3

    cmos = cv2.imread("../read_data/smallmidcrop/cmos_shelf4-1ms.png", -1)
    cmos = np.dstack((cmos, cmos, cmos))
    cmos = cmos / (.001 * .7)

    # plt.imshow(spad / cmos.max())
    # plt.show()
    #
    # plt.imshow(cmos / cmos.max())
    # plt.show()

    h, w, _ = cmos.shape
    cmos_ss = cv2.resize(cmos, (256, 256), interpolation=cv2.INTER_LINEAR)
    spad_ss = cv2.resize(spad, (64, 64), interpolation=cv2.INTER_LINEAR)
    print("cmos saturation = {}".format(cmos_ss.max()))

    plt.imshow(cmos_ss / cmos_ss.max())
    plt.show()
    plt.imshow(spad_ss / spad_ss.max())
    plt.show()
    radiance_writer(cmos_ss, "../read_data/out/cmos_mid_256x256.hdr")
    radiance_writer(spad_ss, "../read_data/out/spad_mid_64x64.hdr")


def super_res():
    cmos = cv2.imread("../read_data/out/cmos.hdr", -1)
    spad = cv2.imread("../read_data/out/spad.hdr", -1)

    h, w, _ = cmos.shape
    print(cmos.shape)
    cmos_ss = cv2.resize(cmos, (256, 256), interpolation=cv2.INTER_LINEAR)
    spad_ss = cv2.resize(spad, (64, 64), interpolation=cv2.INTER_LINEAR)
    print(cmos_ss.shape)
    print(spad_ss.shape)

    plt.imshow(cmos_ss/cmos_ss.max())
    plt.show()
    plt.imshow(spad_ss/spad_ss.max())
    plt.show()
    radiance_writer(cmos_ss, "../read_data/out/cmos_256x256.hdr")
    radiance_writer(spad_ss, "../read_data/out/spad_64x64.hdr")


def main():
    read_mat_fire()
    # super_res()
    # read_small_mid_crop()


if __name__ == "__main__":
    main()
