import cv2
import numpy as np
import matplotlib.pyplot as plt
from simulators.radiance_writer import radiance_writer, radiance_writer_grayscale


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


cmos = cv2.imread("../simulated_outputs/CMOS/3_cmos.png", -1)
ideal = cv2.imread("../simulated_outputs/ideal/3_gt.hdr", -1)
spad = cv2.imread("../simulated_outputs/SPAD/3_spad.hdr", -1)

# cmos_mono = cmos[:, :, 0]
# cv2.imwrite("../simulated_outputs/mono_test.png", cmos_mono)

ideal = cv2.cvtColor(ideal, cv2.COLOR_BGR2RGB)
radiance_writer(ideal, "../simulated_outputs/ideal_rad.hdr")

spad_mono = spad[:, :, 0]
radiance_writer_grayscale(spad_mono, "../simulated_outputs/spad_mono.hdr")

print("hi")