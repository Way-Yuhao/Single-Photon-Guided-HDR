import cv2
import numpy as np
import os
import os.path as p
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
from shutil import copyfile
from radiance_writer import radiance_writer

test_path = "../test/sims/ideal"
train_dev_path = "../simulated_outputs/combined_shuffled/ideal"

test_original_path = "../input/HDR_MATLAB_3x3/"
indoor_original_path = "../input/100samplesDataset"
hdri_original_path = "../simulated_outputs/r_hdri_only"

out_path = "../util/hist/"


def test():
    hdri_range = 469
    if 468 in range(hdri_range):
        print(True)
    else:
        print(False)


def parse_text():
    text_path = "../simulated_outputs/combined_shuffled_recoverable/shuffled.txt"
    recovery_path = "../simulated_outputs/r_hdri_only/"
    shuffled_path = "../simulated_outputs/combined_shuffled_recoverable/ideal"
    txt = open(text_path)
    hdri_range = 469  # [0, 469]
    counter = 0
    for line in txt:
        in_, out_ = [int(x.group()) for x in re.finditer(r'\d+', line)]
        if in_ in range(hdri_range):
            copyfile(p.join(shuffled_path, "{}_gt.hdr".format(out_)), p.join(recovery_path, "{}_gt.hdr".format(out_)))
            counter += 1
    print('detected {} files pertaining to HDRI dataset'.format(counter))


def unscaled():
    recovery_path = "../simulated_outputs/r_hdri_only/"
    unscaled_apth = "../simulated_outputs/hdri_unscaled/"
    factor = 1e6 * 5

    path, dirs, files = next(os.walk(recovery_path))
    file_count = len([x for x in files if "hdr" in x or "exr" in x])
    print("unscaling {} hdr files".format(file_count))

    for filename in tqdm(os.listdir(recovery_path)):
        img = cv2.imread(os.path.join(recovery_path, filename), -1).astype('float64')
        img /= factor
        img = img.astype('float32')
        radiance_writer(img, p.join(unscaled_apth, filename))


def create_hist_single():
    path_img = "../test/sims/ideal/92_gt.hdr"
    img = cv2.imread(path_img, -1)
    log_img = np.log10(img + 1)
    plt.hist(log_img.flatten(), bins=1000, log=True, range=[0, 15])
    # plt.xlim((1, 14))
    plt.xlabel("log(pixel value)")
    plt.ylabel("Number of pixels")
    plt.title("with range, without xlim")
    plt.show()


def compute_bins(dataset_path, title, scaled, scaling_factor=1.0):
    path, dirs, files = next(os.walk(dataset_path))
    file_count = len([x for x in files if "hdr" in x or "exr" in x])
    print("processing {} hdr files".format(file_count))
    nb_bins = 1000
    if scaled is True:
        log10_max_pixel_value = 15
    else:
        log10_max_pixel_value = 10
    if scaling_factor != 1.0:
        print("Warning: attempting to undo a scaling factor of {}".format(scaling_factor))
    count_g = np.zeros(nb_bins)

    hist_g = None

    for filename in tqdm(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, filename), -1)
        if img is None:
            print("ERROR: encountered an empty image named {}".format(filename))
            continue
        if scaling_factor != 1.0:
            img /= scaling_factor
        log_img = np.log10(img + 1)  # 1e-5 or drop all zeros
        # img_temp = img[img > 0]
        hist_g = np.histogram(log_img, bins=nb_bins, range=[0, log10_max_pixel_value])
        count_g += hist_g[0]

    print("saving .npy file to {}".format(p.join(out_path, title + ".npy")))
    np.save(p.join(out_path, "{}_count_.npy".format(title)), count_g)
    np.save(p.join(out_path, "{}_bins_.npy".format(title)), hist_g[1])


def plot_hist_from(title, plt_show=False, scaled=True):
    r_counts = np.load(p.join(out_path, "{}_count_.npy".format(title)))
    r_bins = np.load(p.join(out_path, "{}_bins_.npy".format(title)))
    assert(r_counts is not None and r_bins is not None)

    fig = plt.figure(figsize=(9, 6))
    # plt.bar(r_bins[:-1], r_counts, log=True)
    plt.bar(r_bins[:-1], r_counts, width=np.diff(r_bins), log=True, align="edge")

    plt.title(title)

    if scaled:
        plt.ylim((1, 10e8))
        plt.xlim((-0.5, 13))
    else:
        plt.ylim((1, 10e8))
        plt.xlim((-0.5, 7))

    if plt_show is True:
        plt.show()
    else:
        plt.savefig(p.join(out_path, "{}.svg".format(title)), format="svg", transparent=True, bbox_inches='tight')
        plt.savefig(p.join(out_path, "{}.png".format(title)), format="png")


def test_ideal_wrapper(plt_show):
    title = "test-ideal"
    # compute_bins(dataset_path=test_path, title=title)
    plot_hist_from(title, plt_show=plt_show, scaled=True)


def train_dev_ideal_wrapper(plt_show):
    title = "train-dev-ideal"
    # compute_bins(dataset_path=train_dev_path, title=title)
    plot_hist_from(title, plt_show=plt_show, scaled=True)


def test_original_wrapper(plt_show):
    title = "test-original"
    compute_bins(dataset_path=test_original_path, title=title, scaled=False)
    plot_hist_from(title, plt_show=plt_show, scaled=False)


def indoor_original_wrapper(plt_show):
    title = "indoor-original"
    # compute_bins(dataset_path=indoor_original_path, title=title, scaled=False)
    plot_hist_from(title, plt_show=plt_show, scaled=False)


def hdri_original_wrapper(plt_show):
    title = "hdri-original"
    # compute_bins(dataset_path=hdri_original_path, title=title, scaled=False, scaling_factor=5e6)
    plot_hist_from(title, plt_show=plt_show, scaled=False)


def main():
    plt_show = False
    # train_dev_ideal_wrapper(plt_show)
    # test_ideal_wrapper(plt_show)
    # test_original_wrapper(plt_show)
    indoor_original_wrapper(plt_show)
    # hdri_original_wrapper(plt_show)
    # parse_text()
    # test()
    # unscaled()


if __name__ == "__main__":
    main()
