import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

mpl.use('macosx')
img = cv2.imread("../simulated_outputs/CMOS/0_cmos.hdr", -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/img.max())
plt.show()