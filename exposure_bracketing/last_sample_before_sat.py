import cv2
import numpy as np
import os.path as p
import os
from tqdm import tqdm
from radiance_writer import *

out_path = ""  # define in main()
CMOS_fwc = 33400  # full well capacity of the CMOS sensor
CMOS_T = .01  # exposure time of the CMOS sensor, in seconds
CMOS_sat = CMOS_fwc / CMOS_T  # saturation value of the CMOS simulated images

def add(img1, img2):
    out = img1 + img2
    cv2.imwrite("./naive/naive_merge3.png", (out / 2).astype('uint16'))
    return out


def last_sample_before_sat(img1, img2):
    out = img1
    # print(out.shape[2])
    for h in range(out.shape[0]):
        for w in range(out.shape[1]):
            for c in range(out.shape[2]):
                if out[h][w][c] >= 2**16-1:
                    out[h][w][c] = img2[h][w][c]
    cv2.imwrite("./naive/last_sample2.png", out.astype('uint16'))


def scale(img1, img2, diff):
    return img1, img2 * diff


def apply_gamma(img, g):
    img = np.power(img/float(np.max(img)), 1/g)
    img = img * 2**16
    img[img >= 2**16 - 1] = 2**16 - 1
    return img


def last_sample_before_sat_scaling(img1, img2, diff):
    img1, img2 = scale(img1, img2, diff)
    out = img1
    for h in range(out.shape[0]):
        for w in range(out.shape[1]):
            for c in range(out.shape[2]):
                if out[h][w][c] >= CMOS_sat:  # 2**16 - 1
                    out[h][w] = img2[h][w]
                    continue
    return out


def create_dir(out_path):
    os.mkdir(out_path)


def save(img, idx, mode="png", gamma=2.2):
    if mode == "png":
        img = apply_gamma(img, gamma)
        cv2.imwrite(out_path + "{}_merged.png".format(idx), img.astype('uint16'))
    else:  # hdr
        img = np.dstack((img[:, :, 2], img[:, :, 1], img[:, :, 0]))
        radiance_writer(img, out_path + "{}_merged.hdr".format(idx))
        # b g r to r g b


def run():
    global out_path
    out_path = "../simulated_outputs/artificial/"
    path1 = "../simulated_outputs/artificial/log_out/0_cmos_.000001s.hdr"
    path2 = "../simulated_outputs/artificial/log_out/0_cmos_.01s.hdr"
    # path1 = "../simulated_outputs/artificial/log_out/0_gt_.000001s.hdr"
    # path2 = "../simulated_outputs/artificial/log_out/0_gt_.01s.hdr"


    img2 = cv2.imread(path1, -1).astype('float64')
    img1 = cv2.imread(path2, -1).astype('float64')
    merged = last_sample_before_sat_scaling(img1, img2, 1000000)
    save(merged, 0, "hdr", gamma=4)


def run_all():
    global out_path
    # diff = 10000
    diff = 1
    # path1 is long exposure, path2 is short exposure
    path1 = "../simulated_outputs/combined_shuffled_copy/CMOS/"
    path2 = "../simulated_outputs/combined_shuffled_copy/CMOS_short/"
    out_path = "./out_dev/"
    os.mkdir(out_path)

    path, dirs, files = next(os.walk(path1))
    file_count = len([x for x in files if "hdr" in x])
    print(file_count)

    print("processing {} hdr files".format(file_count))
    for i in tqdm(range(file_count)):
        img1 = cv2.imread(path1 + "{}_cmos.hdr".format(i), -1).astype('float64')
        img2 = cv2.imread(path2 + "{}_cmos.hdr".format(i), -1).astype('float64')
        merged = last_sample_before_sat_scaling(img1, img2, diff)
        save(merged, i, "hdr", gamma=4)

        # only run devs
        if i >= 113:
            break


def cvt_monochrome():
    # fname = p.join(path, "img_1.exr")
    # img = cv2.imread(fname, -1)

    rgb_path = "./out_dev/"
    mono_path = "./out_dev_mono/"

    for i in tqdm(range(112)):
        fname = p.join(rgb_path, "{}_merged.hdr".format(i))
        img = cv2.imread(fname, -1)
        monochrome = np.dstack((img[:, :, 1], img[:, :, 1], img[:, :, 1]))
        out_fname = p.join(mono_path, "{}_merged_monochrome.hdr".format(i))
        radiance_writer(monochrome, out_fname)

    return


def main():
    # run_all()
    # run()
    cvt_monochrome()


if __name__ == "__main__":
    main()
