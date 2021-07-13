import cv2
import numpy as np
import os
import os.path as p
from matplotlib import pyplot as plt
from tqdm import tqdm

ldr_path = "../test/test_baselines/CMOS/"
expand_net_path = "../test/test_baselines/ExpandNet/"
siggraph_2020_path = "../test/test_baselines/SIGGRAPH2020/"
exp_brkt_path = "../test/test_baselines/ExposureBracketing_10kX/"
gt_path = "../test/test_baselines/ground_truth/"
spad_hdr_path = "../test/SPAD-HDR/-v2.15.14-opt2/"

ldr_fnames = "{}_CMOS_monochrome.hdr"
expand_net_fnames = "{}_cmos_prediction_monochrome.hdr"
exp_brkt_2020_fnames = "{}_merged.hdr"
siggraph_2020_fnames = "img_{}.exr"
gt_fnames = "{}_gt_monochrome.hdr"
spad_hdr_fnames = "output-v2.15.14-opt2_{}.hdr"

ldr_tm_path = "../test/test_baselines_tonemapped/CMOS/"
expand_net_tm_path = "../test/test_baselines_tonemapped/ExpandNet/"
siggraph_2020_tm_path = "../test/test_baselines_tonemapped/SIGGRAPH2020/"
exp_brkt_tm_path = "../test/test_baselines_tonemapped/ExposureBracketing_10kX/"
gt_tm_path = "../test/test_baselines_tonemapped/ground_truth/"
spad_hdr_tm_path = "../test/test_baselines_tonemapped/spad_hdr/"

ldr_meta = {
    "name": "cmos",
    "path": ldr_path,
    "fname": ldr_fnames,
    "out_path": ldr_tm_path
}

expand_net_meta = {
    "name": "expand_net",
    "path": expand_net_path,
    "fname": expand_net_fnames,
    "out_path": expand_net_tm_path
}

siggraph_2020_meta = {
    "name": "siggraph_2020",
    "path": siggraph_2020_path,
    "fname": siggraph_2020_fnames,
    "out_path": siggraph_2020_tm_path
}

exp_brkt_meta = {
    "name": "exp_brkt",
    "path": exp_brkt_path,
    "fname": exp_brkt_2020_fnames,
    "out_path": exp_brkt_tm_path
}

gt_meta = {
    "name": "ground_truth",
    "path": gt_path,
    "fname": gt_fnames,
    "out_path": gt_tm_path
}

spad_hdr_meta = {
    "name": "spad_hdr",
    "path": spad_hdr_path,
    "fname": spad_hdr_fnames,
    "out_path": spad_hdr_tm_path
}


def tone_map(img):
    tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
    img = tonemapDrago.process(img)
    return img


def run_all(meta):
    # path, fname, out_path = meta["path"], meta["fname"], meta["out_path"]
    for i in tqdm(range(105)):
        run_single(meta, i, plot=False)


def run_single(meta, idx, plot=False):
    path, fname, out_path = meta["path"], meta["fname"], meta["out_path"]
    img_path = p.join(path, fname.format(idx))
    img = cv2.imread(img_path, -1)
    t_img = tone_map(img)
    if plot:
        plt.imshow(t_img)
        plt.show()
    else:
        t_img = np.nan_to_num(t_img, nan=0.0)
        img_16bit = t_img * 2**16
        img_16bit[img_16bit >= 2 ** 16 - 1] = 2 ** 16 - 1
        img_16bit = img_16bit.astype('uint16')

        out_fname = p.join(out_path, "tm_" + fname.format(idx)[:-3] + "png")
        if not p.exists(out_path):
            os.mkdir(out_path)
        if p.exists(out_fname):
            raise FileExistsError("ERROR: file {} already exists".format(out_fname))
        else:
            cv2.imwrite(out_fname, img_16bit)
        # img /= img.max()


def compare_baselines():


    run_all(ldr_meta)


def vis_real_data():
    scene = "fire"
    path = "../read_data/network_output/{}/".format(scene)
    fname = "output-v2.15.14-opt2_0.hdr"
    img = cv2.imread(p.join(path, fname), -1)
    t_img = tone_map(img)

    t_img = np.nan_to_num(t_img, nan=0.0)
    img_16bit = t_img * 2 ** 16
    img_16bit[img_16bit >= 2 ** 16 - 1] = 2 ** 16 - 1
    img_16bit = img_16bit.astype('uint16')

    out_fname = p.join(p.join(path, "tm_" + fname[:-3] + "png"))
    if p.exists(out_fname):
        raise FileExistsError("ERROR: file {} already exists".format(out_fname))
    else:
        cv2.imwrite(out_fname, img_16bit)


def tone_map_manual():
    input_path = "../read_data/baselines/expand_net/fire_cmos_.5ms_256x256_prediction.hdr"
    output_path = "../read_data/baselines/expand_net/expandnet_fire.png"
    img = cv2.imread(input_path, -1)

    img = np.dstack((img[:, :, 1], img[:, :, 1], img[:, :, 1]))
    t_img = tone_map(img)

    t_img = np.nan_to_num(t_img, nan=0.0)
    img_16bit = t_img * 2 ** 16
    img_16bit[img_16bit >= 2 ** 16 - 1] = 2 ** 16 - 1
    img_16bit = img_16bit.astype('uint16')

    cv2.imwrite(output_path, img_16bit)


def resize_manual():
    input_path = "../read_data/network_output/fire/cmos_Tunnel2-2-.5ms.png"
    output_path = "../read_data/network_output/resized_cmos/fire_cmos_.5ms_64x64.png"
    cmos = cv2.imread(input_path, -1)
    cmos_ss = cv2.resize(cmos, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(cmos_ss.dtype)
    cv2.imwrite(output_path, cmos_ss)


"""=================================================================================="""


def crop_baselines_wrapper():
    # pt = (400, 10) suffix = "A"
    # pt = (490, 1750) suffix = "B"
    # crop_baselines((500, 1360), "C", 3)

    # crop_baselines(103, (400, 1460), "A")
    crop_baselines(103, (710, 860), "B")


def crop_baselines(idx, pt, suffix):
    out_path = "../paper/asset/1-baselines"
    crop_size = (256, 256)


    assert(pt[0] + crop_size[0] < 1024 and pt[1] + crop_size[1] < 2048)

    figure_crop(ldr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(exp_brkt_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(siggraph_2020_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(spad_hdr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(gt_meta, idx, crop_size, pt, out_path, suffix, plot=False)


def figure_crop(meta, idx, size, pt, out_path, suffix, plot=False):
    name, path, fname = meta["name"], meta["out_path"], meta["fname"]
    img_path = p.join(path, "tm_" + fname.format(idx)[:-3] + "png")
    img = cv2.imread(img_path, -1)
    crop = img[pt[0]:pt[0]+size[0], pt[1]:pt[1]+size[1], :]
    if plot:
        plt.imshow(crop / crop.max())
        plt.show()

    out_fname = p.join(out_path, "{}_{}_crop_{}.png".format(name, idx, suffix))
    if p.exists(out_fname):
        raise FileExistsError("ERROR: file {} already exists.".format(out_fname))
    cv2.imwrite(out_fname, crop)


def main():
    # compare_baselines()
    # run_single()
    # vis_real_data()
    # tone_map_manual()
    # resize_manual()
    crop_baselines_wrapper()

if __name__ == "__main__":
    main()
