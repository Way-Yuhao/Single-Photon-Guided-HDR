import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import os
from tqdm import tqdm


img = cv2.imread("../simulated_outputs/CMOS/2_cmos.png", -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2)
plt.imshow(img/img.max())
plt.show()
