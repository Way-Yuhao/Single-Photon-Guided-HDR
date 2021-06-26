import cv2
import numpy as np
import os
from tqdm import tqdm
from radiance_writer import *

out_path = ""  # define in main()


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
                if out[h][w][c] >= 2 ** 16 - 1:
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
    diff = 100
    folder1 = "../simulated_outputs/437_256x128_bl/CMOS/"
    folder2 = "../simulated_outputs/437_256x128_bl/CMOS_x0.01/"
    out_path = "./out/"
    os.mkdir(out_path)

    path, dirs, files = next(os.walk(folder1))
    file_count = len([x for x in files if "png" in x])
    print(file_count)

    print("processing {} hdr files".format(file_count))
    for i in tqdm(range(file_count)):
        img1 = cv2.imread(folder1 + "{}_cmos.png".format(i), -1).astype('float64')
        img2 = cv2.imread(folder2 + "{}_cmos.png".format(i), -1).astype('float64')
        merged = last_sample_before_sat_scaling(img1, img2, diff)
        save(merged, i, "hdr", gamma=4)


def main():
    # run_all()
    run()


if __name__ == "__main__":
    main()