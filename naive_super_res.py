"""
Naive super resolution. Every pixel in the original image is simpled duplicated to a group
of 4x4 pixels of the same value

assuming rgb
"""

import cv2
import numpy as np


_img_path = "./input/starter/low_res.hdr"
_output_path = "./input/starter/high_res.hdr"
# read image
lr_img = cv2.imread(_img_path, -1)  # low resolution
lr_width = len(lr_img[0])
lr_height = len(lr_img)
hr_img = np.zeros((4*lr_height, 4*lr_width, 3), dtype=np.float32)
for i in range(lr_height):
    for j in range(lr_width):
        for c in range(3):
            hr_img[4*i+3][4*j][c], hr_img[4*i+3][4*j+1][c], hr_img[4*i+3][4*j+2][c], hr_img[4*i+3][4*j+3][c] = lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c]
            hr_img[4*i+2][4*j][c], hr_img[4*i+2][4*j+1][c], hr_img[4*i+2][4*j+2][c], hr_img[4*i+2][4*j+3][c] = lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c]
            hr_img[4*i+1][4*j][c], hr_img[4*i+1][4*j+1][c], hr_img[4*i+1][4*j+2][c], hr_img[4*i+1][4*j+3][c] = lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c]
            hr_img[4*i][4*j][c], hr_img[4*i][4*j+1][c], hr_img[4*i][4*j+2][c], hr_img[4*i][4*j+3][c] = lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c], lr_img[i][j][c]  # every lower row
cv2.imwrite(_output_path, hr_img)