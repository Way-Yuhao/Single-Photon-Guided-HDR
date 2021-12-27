import os
import os.path as p
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import scipy.io
from matplotlib import pyplot as plt
from radiance_writer import radiance_writer

path = "../test/sims"
out_path = "../test/CMOS_8bit_gamma_PNG"


def cvt_monochrome_all():
    input_path = "../test/sims/CMOS_8bit_PNG"
    output_path = "../test/sims/CMOS_8bit_mono"
    for i in tqdm(range(105)):
        img = cv2.imread(p.join(input_path, "{}_cmos.png".format(i)))
        img = np.dstack((img[:, :, 1], img[:, :, 1], img[:, :, 1]))
        cv2.imwrite(p.join(output_path, "{}_cmos.png".format(i)), img)


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


def create_8bit_gamma_pngs():
    cmos_path = p.join(path, "CMOS")
    cmos_list = natsorted(os.listdir(cmos_path))
    print(len(cmos_list))
    dataset_max = 3342336.0
    # ratio = dataset_max / 2**8

    length = len(cmos_list)

    for i in tqdm(range(length)):
        fname = "{}_cmos.hdr".format(i)
        cmos = cv2.imread(p.join(cmos_path, fname), -1).astype('float64')
        cmos = cmos / dataset_max  # -> [0, 1]
        cmos = np.sqrt(cmos)  # -> [0, 1] with gamma
        cmos = cmos * 255
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


def read_unwarped():
    in_path = "../read_data/fire/count_img_linearized_warped_translationonly.mat"
    out_fname = "../read_data/out/fire_spad_translation_only.png"
    mat = scipy.io.loadmat(in_path)
    spad = mat["img_data"]
    spad = np.dstack((spad, spad, spad)).astype('float32')
    # spad /= 90

    # plt.imshow(spad / spad.max())
    # plt.show()

    tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
    t_img = tonemapDrago.process(spad)
    t_img = np.nan_to_num(t_img, nan=0.0)
    img_16bit = t_img * 2 ** 16
    img_16bit[img_16bit >= 2 ** 16 - 1] = 2 ** 16 - 1
    img_16bit = img_16bit.astype('uint16')
    cv2.imwrite(out_fname, img_16bit)


def read_gt():
    in_path = "../read_data/fire/cmos_Tunnel2-2_hdr_img.mat"
    out_fname = "../read_data/out/fire_gt.png"
    mat = scipy.io.loadmat(in_path)
    gt = mat["img_data"]
    gt = np.dstack((gt, gt, gt)).astype('float32')
    # spad /= 90

    plt.imshow(gt / gt.max())
    plt.show()

    tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
    t_img = tonemapDrago.process(gt)
    t_img = np.nan_to_num(t_img, nan=0.0)
    img_16bit = t_img * 2 ** 16
    img_16bit[img_16bit >= 2 ** 16 - 1] = 2 ** 16 - 1
    img_16bit = img_16bit.astype('uint16')
    cv2.imwrite(out_fname, img_16bit)


def main():
    # read_mat_fire()
    # super_res()
    # read_small_mid_crop()

    # cvt_monochrome_all()
    # read_gt()
    create_8bit_gamma_pngs()

if __name__ == "__main__":
    main()
