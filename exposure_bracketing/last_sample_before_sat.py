import cv2
import numpy as np


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
    return img1, img2 * 10


def apply_gamma(img):
    img = np.power(img/float(np.max(img)), 1/2.2)
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
    cv2.imwrite("./naive/last_sample_scaling.png", out.astype('uint16'))


def main():
    img1 = cv2.imread("./exp_brkt/long_cmos.png", -1).astype('float64')
    img2 = cv2.imread("./exp_brkt/short_cmos.png", -1).astype('float64')
    last_sample_before_sat_scaling(img1, img2)


if __name__ == "__main__":
    main()