import numpy as np
import cv2
_path = "./input/simulator/"
a = cv2.imread(_path + "yes.hdr", -1)
b = np.load(_path + "photon_count.npy")
print("what")