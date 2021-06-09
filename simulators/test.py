import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import os
from tqdm import tqdm


def rand_horizontal_flip(img, p=.5):
    x = np.random.rand()
    if x > p:
        return np.flip(img, axis=1)
    else:
        return img


img = cv2.imread("../simulated_outputs/artificial/scene_gray_text.hdr", -1)
img = rand_horizontal_flip(img)

plt.imshow(img/img.max())
plt.show()
