import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp
a = 1
print(exp(-a))

path = "../input/100samplesDataset/9C4A0034-a460e29cd9.exr"

img = cv2.imread(path, -1)


print(img)
