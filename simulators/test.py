import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import os.path as p

# mpl.use('macosx')
# img = cv2.imread("../simulated_outputs/CMOS/0_cmos.hdr", -1)
# print(img.shape)

# cmos = cv2.imread("../simulated_outputs/combined_shuffled/CMOS/300_cmos.hdr", -1)
# spad = cv2.imread("../simulated_outputs/combined_shuffled/SPAD/300_spad.hdr", -1)
# print(cmos)

out_path = "../test/CMOS/"

fname = p.join(out_path, "0_cmos.hdr")

img = cv2.imread(fname, -1)
print(img.shape)
h, w, _ = img.shape
if h % 2 != 0:
    img = img[:-1, :, :]
if w % 2 != 0:
    img = img[:, :-1, :]

print(img.shape)

