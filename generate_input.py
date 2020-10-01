"""

"""

import cv2
import numpy as np

_path = "./input/starter/"
hdr_img = cv2.imread(_path + "hdr_ground_truth.hdr", -1)
# hdr_img = np.array()
assert hdr_img is not None
height, width = hdr_img[:, :, 0].shape
dr = hdr_img.max() / hdr_img.min()
print("original image is {} by {}".format(height, width))
print("dynamic range = ", dr)
cv2.imwrite(_path + "test.png", hdr_img)

# generating low-res image
print("\ngenerating low res image")
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

# generating ldr image
# reduce
print("\ngenerating ldr image")
# ldr_img = hdr_img[::4, ::4, :]
ldr_img = hdr_img.copy()
ub = (ldr_img.max() - ldr_img.min()) * .00001 + ldr_img.min()
t = height * width
c = np.count_nonzero(ldr_img >= ub)
print("count = ", c)
ldr_img[ldr_img >= ub] = ub
ldr_dr = ldr_img.max() / ldr_img.min()
print("dynamic range = ", ldr_dr)
print("dynamic range reduced to ", ldr_dr / dr, "of the original")
cv2.imwrite(_path + "ldr.hdr", ldr_img)
# ldr_temp = cv2.imread(_path + "ldr.hdr", -1)

ldr_img = (ldr_img - ldr_img.min()) / (ldr_img.max() - ldr_img.min())
ldr_img *= 2**16
ldr_img = ldr_img.astype(np.uint16)
ldr_img[ldr_img <= 0] = 2**16 - 1

cv2.imwrite(_path + "ldr.png", ldr_img)

test = hdr_img
test = (test - test.min()) / (test.max() - test.min())
test *= 2**16
test = test.astype(np.uint16)
test[test <= 0] = 2**16 - 1

cv2.imwrite(_path + "test.png", test)
