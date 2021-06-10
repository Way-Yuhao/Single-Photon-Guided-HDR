import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from radiance_writer import radiance_writer
from os import path as p

path = "../simulated_outputs/artificial"


def c0():
    fwc = np.floor(.01 / 150e-9)
    scene = np.logspace(1, 10, num=2048)
    for i in range(10):
        scene = np.vstack((scene, scene))

    img = np.dstack((scene, scene, scene))
    plt.imshow(img / img.max())
    plt.show()
    radiance_writer(img, os.path.join(path, "log_scene.hdr"))
    print(img.min())
    print(img.max())


def add_text():
    img = cv2.imread(p.join(path, "log_scene.hdr"), -1)
    text = cv2.imread(p.join(path, "text.png"), -1)
    val = img.max()
    text = np.dstack((text[:, :, 0], text[:, :, 1], text[:, :, 2]))
    # print(text.shape)
    # plt.imshow(text)
    # plt.show()
    # print(text.min())
    # print(text.max())
    img[text > 0] = val
    plt.imshow(img/img.max())
    plt.show()
    radiance_writer(img, os.path.join(path, "scene_white_text_log.hdr"))


def c2():
    img = cv2.imread(p.join(path, "linear_scene.hdr"), -1)
    print(img.max())


def test():
    img = cv2.imread(p.join(path, "input", "scene_white_text_log.hdr"), -1)
    print(img.max())

def main():
    # construct()
    # c0()
    # add_text()
    test()


if __name__ == "__main__":
    main()
