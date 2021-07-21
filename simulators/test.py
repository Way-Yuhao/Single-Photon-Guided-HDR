import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import os.path as p

cmos_saturation = 33400 / .01


def normalize(img):
    img = img / cmos_saturation * 255
    return img


def main():
    scene = 8
    ours = cv2.imread("../test/SPAD-HDR/-v2.15.14-opt2/output-v2.15.14-opt2_{}.hdr".format(scene), -1)
    no_attn = cv2.imread("../ablation_study/no-attn/output_{}.hdr".format(scene), -1)
    no_spad = cv2.imread("../ablation_study/no-spad/output_{}.hdr".format(scene), -1)
    cmos = cv2.imread("../test/test_baselines/CMOS/{}_CMOS_monochrome.hdr".format(scene), -1)
    cmos = normalize(cmos)

    ours[ours == 0] = ours[ours > 0].min()
    no_attn[no_attn == 0] = no_attn[no_attn > 0].min()
    no_spad[no_spad == 0] = no_spad[no_spad > 0].min()

    cmos_max = np.log10(cmos.flatten()).max()
    no_attn_max = np.log10(no_attn.flatten()).max()
    no_spad_max = np.log10(no_spad.flatten()).max()


    # plt.hist(np.log10(ours.flatten()), bins=1000, alpha=0.5, color='red', label="ours")
    plt.hist(np.log10(no_attn.flatten()), bins=1000, alpha=0.5, color='orange', label="no attn")
    plt.hist(np.log10(no_spad.flatten()), bins=1000, alpha=0.5, color='blue', label="no spad")
    plt.hist(np.log10(cmos.flatten()), bins=1000, alpha=0.5, color='green', label="cmos")
    plt.title("Scene {} | max: cmos={:.3f}, no-spad={:.3f}, no-attn={:.3f}".format(scene, cmos_max, no_spad_max, no_attn_max))
    plt.legend()
    plt.ylim((0, 20000))
    plt.xlim(2.2, 2.6)
    plt.show()



if __name__ == "__main__":
    main()