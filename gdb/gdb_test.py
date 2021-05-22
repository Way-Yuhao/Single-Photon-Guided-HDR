import cv2
from radiance_writer import *
from matplotlib import pyplot as plt

img = cv2.imread("./out/blended.hdr", -1)

# img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
radiance_writer(img, "./out/blended_rgb.hdr")

# plt.imshow(img2/img2.max())
# plt.show()