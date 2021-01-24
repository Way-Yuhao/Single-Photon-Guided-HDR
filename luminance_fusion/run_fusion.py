"""
This module runs lumiance_fusion.py on a collection of images.
Note:
    * Expects SPAD files in .hdr and CMOS in .png
"""

import os
import numpy as np
import cv2
from tqdm import trange
from luminance_fusion import rescale, fusion
from simulators.radiance_writer import radiance_writer

CMOS_dir_path = "../simulated_outputs/CMOS/"
CMOS_suffix = "_cmos.png"
CMOS_ftype = "png"
# SPAD_dir_path = "../simulated_outputs/SPAD_HDR_SR/"
# SPAD_suffix = "_spad_bilinear.hdr"
# SPAD_ftype = "hdr"
# out_dir_path = "../fusion_results/"  # relative path to radiance_writer.py

SPAD_dir_path = "../simulated_outputs/CMOS_mid/"
SPAD_suffix = "_cmos.png"
SPAD_ftype = "png"
out_dir_path = "../exposure_bracketing/long+mid/"  # relative path to radiance_writer.py


def save_img(img, id):
    img_32 = img.astype(np.float32)
    img_32 = cv2.cvtColor(img_32, cv2.COLOR_BGR2RGB)
    radiance_writer(img_32, out_dir_path + str(id) + "_fusion.hdr")


def main():
    _, _, CMOS_files = next(os.walk(CMOS_dir_path))
    CMOS_file_count = len([x for x in CMOS_files if CMOS_ftype in x])
    _, _, SPAD_files = next(os.walk(SPAD_dir_path))
    SPAD_file_count = len([x for x in SPAD_files if SPAD_ftype in x])
    assert(CMOS_file_count == SPAD_file_count)
    print("running luminance fusion on {} pairs of images".format(CMOS_file_count))

    for id in trange(CMOS_file_count):
        ldr_img = cv2.imread(CMOS_dir_path + str(id) + CMOS_suffix, -1)
        hdr_img = cv2.imread(SPAD_dir_path + str(id) + SPAD_suffix, -1)
        ldr_img, hdr_img = rescale(ldr_img, hdr_img)
        fused_img = fusion(ldr_img, hdr_img)
        save_img(fused_img, id)

if __name__ == "__main__":
    main()

