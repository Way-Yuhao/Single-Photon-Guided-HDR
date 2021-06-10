import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import os
from tqdm import tqdm



img = cv2.imread("../simulated_outputs/CMOS/0_cmos.hdr", -1)
img2 = cv2.imread("../simulated_outputs/SPAD/0_spad.hdr", -1)
print(img.median())
print(img2.median())
print(0)