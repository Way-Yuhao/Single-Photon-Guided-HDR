import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np

mpl.use('macosx')
img = cv2.imread("../simulated_outputs/CMOS/0_cmos.hdr", -1)
print(img.shape)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/img.max())
# plt.show()


# q = {                 # quantum efficiency index for each color channel
#     'r': .40,
#     'g': .75,
#     'b': .77
# }
#
# qe = np.array([q["b"], q["g"], q["r"]])
#
#
# p = np.array([[1, 1, 1]])
# print(p * qe)
# sat = p / qe
print(sat)