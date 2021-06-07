import os
from os import path as p
import cv2
import numpy as np

path_1 = "../simulated_outputs/collection/HDRI"
path_2 = "../simulated_outputs/collection/indoor"
path_3 = "../simulated_outputs/collection/MATLAB_3x3"
path_target = "../simulated_outputs/combined"
path_shuf = "../simulated_outputs/combined_shuffled"

composition = ("CMOS", "SPAD", "ideal", "plt")


def check_files(f):
    if os.path.exists(f):
        return True
    else:
        return False


def combine(global_counter, folder):
    counter = 0
    while True:
        cmos = p.join(folder, composition[0], "{}_cmos.png".format(counter))
        spad = p.join(folder, composition[1], "{}_spad.hdr".format(counter))
        ideal = p.join(folder, composition[2], "{}_gt.hdr".format(counter))
        plt = p.join(folder, composition[3], "plt_{}.png".format(counter))
        b1 = check_files(cmos)
        b2 = check_files(spad)
        b3 = check_files(ideal)
        b4 = check_files(plt)
        if not (b1 and b2 and b3 and b4):
            print("counter stopped at {} in folder {}".format(counter, folder))
            print("global counter = ", global_counter)
            break
        else:
            os.rename(cmos, p.join(path_target, "CMOS", "{}_cmos.png".format(global_counter)))
            os.rename(spad, p.join(path_target, "SPAD", "{}_spad.hdr".format(global_counter)))
            os.rename(ideal, p.join(path_target, "ideal", "{}_gt.hdr".format(global_counter)))
            os.rename(plt, p.join(path_target, "plt", "plt_{}.png".format(global_counter)))
            # print(p.join(path_target, "CMOS", "{}_cmos.png".format(global_counter)))

        counter += 1
        global_counter += 1

    return global_counter


def shuffle():
    count = 668
    perm = np.random.permutation(count)
    seq_idx = 0
    for i in perm:
        cmos = p.join(path_target, composition[0], "{}_cmos.png".format(seq_idx))
        spad = p.join(path_target, composition[1], "{}_spad.hdr".format(seq_idx))
        ideal = p.join(path_target, composition[2], "{}_gt.hdr".format(seq_idx))
        plt = p.join(path_target, composition[3], "plt_{}.png".format(seq_idx))

        cmos_shuff = p.join(path_shuf, composition[0], "{}_cmos.png".format(i))
        spad_shuff = p.join(path_shuf, composition[1], "{}_spad.hdr".format(i))
        ideal_shuff = p.join(path_shuf, composition[2], "{}_gt.hdr".format(i))
        plt_shuff = p.join(path_shuf, composition[3], "plt_{}.png".format(i))

        os.rename(cmos, cmos_shuff)
        os.rename(spad, spad_shuff)
        os.rename(ideal, ideal_shuff)
        os.rename(plt, plt_shuff)
        seq_idx += 1


def test():
    print(np.random.permutation(10))


def main():
    # counter = 0
    # counter = combine(counter, path_1)
    # counter = combine(counter, path_2)
    # counter = combine(counter, path_3)
    shuffle()


if __name__ == "__main__":
    main()
