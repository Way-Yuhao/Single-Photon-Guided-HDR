import cv2
import numpy as np
import os
import os.path as p
from matplotlib import pyplot as plt
from tqdm import tqdm
from radiance_writer import radiance_writer

ldr_path = "../test/test_baselines/CMOS/"
expand_net_path = "../test/test_baselines/ExpandNet/"
siggraph_2020_path = "../test/test_baselines/rgb/SIGGRAPH2020-rgb"
exp_brkt_path = "../test/test_baselines/rgb/ExpandNet-rgb"
gt_path = "../test/test_baselines/ground_truth/"
spad_hdr_path = "../test/SPAD-HDR/-v2.15.14-opt2/"

ldr_fnames = "{}_CMOS_monochrome.hdr"
expand_net_fnames = "{}_cmos_prediction.hdr"
exp_brkt_2020_fnames = "{}_merged.hdr"
siggraph_2020_fnames = "img_{}.exr"
gt_fnames = "{}_gt_monochrome.hdr"
spad_hdr_fnames = "output-v2.15.14-opt2_{}.hdr"

ldr_tm_path = "../test/test_baselines_tonemapped/CMOS/"
expand_net_tm_path = "../test/test_baselines_tonemapped/rgb/ExpandNet-RGB-input"
siggraph_2020_tm_path = "../test/test_baselines_tonemapped/rgb/SIGGRAPH2020-RGB-input"
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

esrgan_meta = {
    "name": "msrgan",
    "path": "../test/test_baselines/ESRGAN",
    "fname": "{}_SPAD_ESRGAN_4x.hdr",
    "out_path": "../test/test_baselines_tonemapped/esrgan/"
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

spad_meta = {
    "name": "spad",
    "path": "../test/sims/SPAD",
    "fname": "{}_spad.hdr",
    "out_path": "../test/test_baselines_tonemapped/spad/"
}

laplacian_meta = {
    "name": "laplacian",
    "path": "../test/test_baselines/Laplacian",
    "fname": "{}_laplacian.hdr",
    "out_path": "../test/test_baselines_tonemapped/Laplacian/"
}

##########################################################################################

path = "../exposure_bracketing/snr_study/out/"

exp_brkt_dev_10kx_meta = {
    "name": "exp_brkt_dev_10kx",
    "path": "../dev/baselines_monochrome/exp_brkt_10000x_monochrome",
    "fname": "{}_merged_monochrome.hdr",
    "out_path": "../dev/tonemapped_baselines/exp_brkt_10kx"
}


exp_brkt_14_meta = {
    "name": "exp_brkt_2x",
    "path": None,
    "fname": "{}_2x_merged.hdr",
    "out_path": "../exposure_bracketing/snr_study/out/"
}

spad_hdr_dev_meta = {
    "name": "sapd_hdr_dev",
    "path": "../dev/baselines_monochrome/spad_hdr",
    "fname": "output_{}.hdr",
    "out_path": "../dev/tonemapped_baselines/spad_hdr"
}

path_temp = "../dev/sims/ideal_dev"

gt_dev_meta = {
    "name": "gt_hdr_dev",
    "path": "../dev/sims/ideal_dev",
    "fname": "{}_gt.hdr",
    "out_path": "../dev/tonemapped_baselines/gt"
}

######################################################################################

ablation_no_attn_meta = {
    "name": "ablation_no_attn",
    "path": "../ablation_study/no-attn/",
    "fname": "output_{}.hdr",
    "out_path": "../ablation_study/no-attn_tm/"
}

ablation_no_spad_meta = {
    "name": "ablation_no_spad",
    "path": "../ablation_study/no-spad/",
    "fname": "output_{}.hdr",
    "out_path": "../ablation_study/no-spad_tm/"
}



def tone_map(img):
    tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
    img = tonemapDrago.process(img)
    return img


def run_all(meta, mode="test", extract_ch=False):
    if mode == "test":
        size = 105
    elif mode == "dev":
        size = 111
    else:
        raise ValueError("Mode can be either test or dev")
    for i in tqdm(range(size)):
        run_single(meta, i, plot=False, extract_ch=extract_ch)


def run_single(meta, idx, plot=False, extract_ch=False):
    path, fname, out_path = meta["path"], meta["fname"], meta["out_path"]
    img_path = p.join(path, fname.format(idx))
    img = cv2.imread(img_path, -1)

    if extract_ch:
        img = np.dstack((img[:, :, 1], img[:, :, 1], img[:, :, 1]))

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


def cvt_monocrhome_manual():
    input_path = "../dev/sims/ideal_dev/14_gt.hdr"
    output_path = "../paper/asset/gt_14_mono.hdr"
    img = cv2.imread(input_path, -1)
    img = np.dstack((img[:, :, 1], img[:, :, 1], img[:, :, 1]))
    cv2.imwrite(output_path, img)


def resize_manual():
    input_path = "../read_data/network_output/fire/cmos_Tunnel2-2-.5ms.png"
    output_path = "../read_data/network_output/resized_cmos/fire_cmos_.5ms_64x64.png"
    cmos = cv2.imread(input_path, -1)
    cmos_ss = cv2.resize(cmos, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(cmos_ss.dtype)
    cv2.imwrite(output_path, cmos_ss)


####################################################################################

def crop_snr_wrapper():
    crop_snr(14, (400, 2150), "A")
    # crop_snr(14, (900, 2400), "B")
    crop_snr(14, (900, 2400), "C", crop_size=(512, 512))


def crop_snr(idx, pt, suffix, crop_size=(256, 256)):
    out_path = "../paper/asset/2-snr"
    # crop_size = (256, 256)

    assert (pt[0] + crop_size[0] < 2048 and pt[1] + crop_size[1] < 4096)

    # figure_crop(ldr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(exp_brkt_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(siggraph_2020_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(expand_net_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(spad_hdr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(gt_meta, idx, crop_size, pt, out_path, suffix, plot=False)

    # figure_crop(exp_brkt_dev_10kx_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(spad_hdr_dev_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(gt_dev_meta, idx, crop_size, pt, out_path, suffix)

    figure_crop(exp_brkt_14_meta, idx, crop_size, pt, out_path, suffix, plot=False)


def figure_crop_hdr(meta, idx, size, pt, out_path, suffix, plot=False, r=1):

    if r != 1:
        size = (int(size[0] / r), int(size[1] / r))
        pt = (int(pt[0] / r), int(pt[1] / r))

    name, path, fname = meta["name"], meta["path"], meta["fname"]
    img_path = p.join(path, fname.format(idx))
    img = cv2.imread(img_path, -1)
    crop = img[pt[0]:pt[0]+size[0], pt[1]:pt[1]+size[1], :]
    if plot:
        plt.imshow(crop / crop.max())
        plt.show()

    out_fname = p.join(out_path, "{}_{}_crop_{}.hdr".format(name, idx, suffix))
    if p.exists(out_fname):
        raise FileExistsError("ERROR: file {} already exists.".format(out_fname))
    radiance_writer(crop, out_fname)


def crop_ablation_wrapper():
    crop_ablation(3, (400, 10), "A")
    crop_ablation(3, (500, 1360), "B")

    crop_ablation(103, (710, 860), "B")
    crop_ablation(103, (300, 1460), "C")
    #
    crop_ablation(24, (320, 600), "A")
    crop_ablation(24, (750, 450), "B")

    crop_ablation(25, (190, 860), "A")
    crop_ablation(25, (850, 1250), "C", crop_size=(128, 128))

    crop_ablation(11, (60, 800), "A")
    crop_ablation(11, (580, 930), "B")


def crop_ablation(idx, pt, suffix, crop_size=(256, 256)):
    out_path = "../paper/asset/10[sup]-ablation"
    assert (pt[0] + crop_size[0] < 1024 and pt[1] + crop_size[1] < 2048)

    # figure_crop(ldr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(siggraph_2020_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(spad_hdr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(gt_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(ablation_no_spad_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    # figure_crop(ablation_no_attn_meta, idx, crop_size, pt, out_path, suffix, plot=False)


def crop_baselines_wrapper():
    crop_baselines(3, (400, 10), "A")
    crop_baselines(3, (500, 1360), "B")

    # crop_baselines(103, (400, 1460), "A")
    crop_baselines(103, (710, 860), "B")
    crop_baselines(103, (300, 1460), "C")

    crop_baselines(24, (320, 600), "A")
    crop_baselines(24, (750, 450), "B")

    crop_baselines(25, (190, 860), "A")
    # crop_baselines(25, (720, 500), "B")
    crop_baselines(25, (850, 1250), "C", crop_size=(128, 128))

    crop_baselines(11, (60, 800), "A")
    crop_baselines(11, (580, 930), "B")
    return


def crop_baselines(idx, pt, suffix, crop_size=(256, 256)):
    out_path = "../paper/asset/1-baselines"

    assert(pt[0] + crop_size[0] < 1024 and pt[1] + crop_size[1] < 2048)

    figure_crop(ldr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(exp_brkt_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(siggraph_2020_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(expand_net_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(spad_hdr_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(gt_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(spad_meta, idx, crop_size, pt, out_path, suffix, plot=False, r=4)
    figure_crop(esrgan_meta, idx, crop_size, pt, out_path, suffix, plot=False)
    figure_crop(laplacian_meta, idx, crop_size, pt, out_path, suffix, plot=False)


def figure_crop(meta, idx, size, pt, out_path, suffix, plot=False, r=1):

    if r != 1:
        size = (int(size[0] / r), int(size[1] / r))
        pt = (int(pt[0] / r), int(pt[1] / r))

    name, path, fname = meta["name"], meta["out_path"], meta["fname"]
    img_path = p.join(path, "tm_" + fname.format(idx)[:-3] + "png")
    img = cv2.imread(img_path, -1)
    crop = img[pt[0]:pt[0]+size[0], pt[1]:pt[1]+size[1], :]
    if plot:
        plt.imshow(crop / crop.max())
        plt.show()

    out_fname = p.join(out_path, "{}_{}_crop_{}.png".format(name, idx, suffix))
    if p.exists(out_fname):
        # raise FileExistsError("ERROR: file {} already exists.".format(out_fname))
        pass
    cv2.imwrite(out_fname, crop)

####################################################################################


def create_hist():
    path_img = "../dev/sims/ideal_dev/14_gt.hdr"
    img = cv2.imread(path_img, -1)
    log_img = np.log10(img + 1)
    plt.hist(log_img.flatten(), bins=1000, log=True)
    plt.xlim((2.5, 11.5))
    plt.xlabel("log(pixel value)")
    plt.ylabel("Number of pixels")
    plt.show()

    # out_path = "../paper/asset/scene_14_log_hist.svg"
    # plt.savefig(out_path, format="svg")


def create_radiance_map():
    path_1 = "../dev/sims/ideal_dev"
    path_2 = "../dev/baselines_monochrome/spad_hdr/"

    ideal = cv2.imread(p.join(path_1, "14_gt.hdr"), -1)
    spad_hdr = cv2.imread(p.join(path_2, "output_14.hdr"), -1)
    print(ideal.max())
    print(spad_hdr.max())
    spad_hdr = spad_hdr[:, :, 1]
    plt.imshow(spad_hdr / spad_hdr.max())
    plt.show()


def main():
    # compare_baselines()
    # run_single()
    # vis_real_data()
    # tone_map_manual()
    # resize_manual()
    # crop_baselines_wrapper()

    # run_all(expand_net_meta, mode="test", extract_ch=True)

    # create_radiance_map()
    # create_hist()
    # cvt_monocrhome_manual()

    # crop_snr_wrapper()

    crop_ablation_wrapper()

if __name__ == "__main__":
    main()
