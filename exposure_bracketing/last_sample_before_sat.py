import cv2
import numpy as np
import os
from tqdm import tqdm

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


def scale(img1, img2):
    return img1, img2 * 100


def apply_gamma(img):
    img = np.power(img/float(np.max(img)), 1/4)
    img = img * 2**16
    img[img >= 2**16 - 1] = 2**16 - 1
    return img


def last_sample_before_sat_scaling(img1, img2):
    img1, img2 = scale(img1, img2)
    out = img1
    for h in range(out.shape[0]):
        for w in range(out.shape[1]):
            for c in range(out.shape[2]):
                if out[h][w][c] >= 2 ** 16 - 1:
                    out[h][w] = img2[h][w]
                    continue
    out = apply_gamma(out)
    # cv2.imwrite("./naive/last_sample_scaling.png", out.astype('uint16'))
    return out


def create_dir(out_path):
    os.mkdir(out_path)


# def run():
#     img1 = cv2.imread("./exp_brkt/long_cmos.png", -1).astype('float64')
#     img2 = cv2.imread("./exp_brkt/short_cmos.png", -1).astype('float64')
#     last_sample_before_sat_scaling(img1, img2)


def run_all():
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
        merged = last_sample_before_sat_scaling(img1, img2)
        cv2.imwrite(out_path + "{}_merged.png".format(i), merged.astype('uint16'))


def main():
    run_all()


if __name__ == "__main__":
    main()