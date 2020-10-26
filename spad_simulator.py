"""
This module is designed to simulate SPAD photon-counting output given a 32-bit ground truth scene
"""

import cv2
import numpy as np


_path = "./input/starter/"

hdr_img = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
assert hdr_img is not None
height, width = hdr_img[:, :, 0].shape
dr = hdr_img.max() / hdr_img.min()
print("original image is {} by {}".format(height, width))
print("dynamic range = ", dr)
cv2.imwrite(_path + "test.png", hdr_img)

low_res_img = hdr_img.copy()[::4, ::4, :]
lr_height, lr_width = low_res_img[:, :, 0].shape
print("resolution = {} x {}".format(lr_height, lr_width))
# low_res_img = (low_res_img - low_res_img.min()) / (low_res_img.max() - low_res_img.min())
# low_res_img *= 2**16
# low_res_img = low_res_img.astype(np.uint16)
# low_res_img[low_res_img <= 0] = 2**16 - 1
# print(low_res_img.max())
# cv2.imshow('image', low_res_img)
cv2.imwrite(_path + "low_res.hdr", low_res_img)
cv2.imwrite(_path + "low_res.png", low_res_img)# ldr_temp = cv2.imread(_path + "ldr.hdr", -1)


# test = hdr_img
# test = (test - test.min()) / (test.max() - test.min())
# test *= 2**16
# test = test.astype(np.uint16)
# test[test <= 0] = 2**16 - 1
#
# cv2.imwrite(_path + "test.png", test)

def read_img():
    hdr_img = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
    assert hdr_img is not None
    height, width = hdr_img[:, :, 0].shape
    dr = hdr_img.max() / hdr_img.min()
    print("original image is {} by {}".format(height, width))
    print("dynamic range = ", dr)


def down_sample():
    pass




def main():


if __name__ == "__main__":
    main()