import cv2
import numpy as np

cmos = cv2.imread("../simulated_outputs/test/sim/0_cmos.png", -1)
spad = cv2.imread("../simulated_outputs/test/sim/0_gt.hdr", -1)
ideal = cv2.imread("../simulated_outputs/test/sim/0_spad.hdr", -1)

print("hi")