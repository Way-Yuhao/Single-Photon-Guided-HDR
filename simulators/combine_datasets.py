import os
from os import path as p
import cv2
import numpy as np
import re
from shutil import copyfile

path_1 = "../simulated_outputs/collection_short_exp/HDRI"
path_2 = "../simulated_outputs/collection_short_exp/indoor"
path_3 = "../simulated_outputs/collection_short_exp/MATLAB_3x3"
path_combined = "../simulated_outputs/combined_short_exp"
path_shuf = "../simulated_outputs/combined_shuffled_short_exp"

composition = ("CMOS", "SPAD", "ideal", "plt")

count = 561


def check_files(f):
    if os.path.exists(f):
        return True
    else:
        return False


def combine(global_counter, folder):
    counter = 0
    while True:
        cmos = p.join(folder, composition[0], "{}_cmos.hdr".format(counter))
        # spad = p.join(folder, composition[1], "{}_spad.hdr".format(counter))
        # ideal = p.join(folder, composition[2], "{}_gt.hdr".format(counter))
        # plt = p.join(folder, composition[3], "plt_{}.png".format(counter))
        b1 = check_files(cmos)
        # b2 = check_files(spad)
        # b3 = check_files(ideal)
        # b4 = check_files(plt)
        if not (b1):  #  and b2 and b3 and b4):
            print("counter stopped at {} in folder {}".format(counter, folder))
            print("global counter = ", global_counter)
            break
        else:
            os.rename(cmos, p.join(path_combined, "CMOS", "{}_cmos.hdr".format(global_counter)))
            # os.rename(spad, p.join(path_combined, "SPAD", "{}_spad.hdr".format(global_counter)))
            # os.rename(ideal, p.join(path_combined, "ideal", "{}_gt.hdr".format(global_counter)))
            # os.rename(plt, p.join(path_combined, "plt", "plt_{}.png".format(global_counter)))
            print("{} -> {}".format(counter, global_counter))

        counter += 1
        global_counter += 1

    return global_counter


def shuffle():
    perm = np.random.permutation(count)
    seq_idx = 0
    for i in perm:
        cmos = p.join(path_combined, composition[0], "{}_cmos.hdr".format(seq_idx))
        spad = p.join(path_combined, composition[1], "{}_spad.hdr".format(seq_idx))
        ideal = p.join(path_combined, composition[2], "{}_gt.hdr".format(seq_idx))
        plt = p.join(path_combined, composition[3], "plt_{}.png".format(seq_idx))

        cmos_shuff = p.join(path_shuf, composition[0], "{}_cmos.hdr".format(i))
        spad_shuff = p.join(path_shuf, composition[1], "{}_spad.hdr".format(i))
        ideal_shuff = p.join(path_shuf, composition[2], "{}_gt.hdr".format(i))
        plt_shuff = p.join(path_shuf, composition[3], "plt_{}.png".format(i))

        os.rename(cmos, cmos_shuff)
        os.rename(spad, spad_shuff)
        os.rename(ideal, ideal_shuff)
        os.rename(plt, plt_shuff)
        print("{} -> {}".format(seq_idx, i))
        seq_idx += 1


def shuffle_fixed():
    f = open("../simulated_outputs/shuffled_test.txt", "r")
    for i in range(count):
        line = f.readline()
        a, b = re.findall(r'\d+', line)
        cmos = p.join(path_combined, composition[0], "{}_cmos.hdr".format(a))
        cmos_shuff = p.join(path_shuf, composition[0], "{}_cmos.hdr".format(b))
        copyfile(cmos, cmos_shuff)
        print("{} -> {}".format(a, b))


def test():
    f = open("../simulated_outputs/shuffled_test.txt", "r")
    for i in range(count):
        line = f.readline()
        a, b = re.findall(r'\d+', line)
        print(b)



def init():
    out_path = "../simulated_outputs/"
    if p.exists(path_combined) or p.exists(path_shuf):
        raise FileExistsError("ERROR: directory shuffled or combined already exists. Please remove.".format(out_path))

    os.mkdir(path_combined)
    os.mkdir(p.join(path_combined, "CMOS"))
    os.mkdir(p.join(path_combined, "SPAD"))
    os.mkdir(p.join(path_combined, "ideal"))
    os.mkdir(p.join(path_combined, "plt"))

    os.mkdir(path_shuf)
    os.mkdir(p.join(path_shuf, "CMOS"))
    os.mkdir(p.join(path_shuf, "SPAD"))
    os.mkdir(p.join(path_shuf, "ideal"))
    os.mkdir(p.join(path_shuf, "plt"))
    return


def main():
    # init()
    # counter = 0
    # counter = combine(counter, path_1)
    # counter = combine(counter, path_2)
    # shuffle()

    shuffle_fixed()

    # test()


if __name__ == "__main__":
    main()
