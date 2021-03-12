import cv2
import numpy as np
import matplotlib.pyplot as plt


def disp_plt(img, title="", normalize=False):
    """
    :param img: image to display
    :param title: title of the figure
    :param path: path to save the figure. If empty or None, this function will not save the figure
    :param normalize: set to True if intend to normalize the image to [0, 1]
    :return: None
    """
    img = img / img.max() if normalize else img
    plt.imshow(img)
    plt.title(title)
    plt.show()
    return


# cmos = cv2.imread("../simulated_outputs/test/sim/0_cmos.png", -1)
ideal = cv2.imread("../simulated_outputs/test/sim/0_gt.hdr", -1)
# spad = cv2.imread("../simulated_outputs/test/sim/0_spad.hdr", -1)

print("hi")