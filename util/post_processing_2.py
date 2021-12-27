import cv2
import numpy as np
import os
import os.path as p
from matplotlib import pyplot as plt
from tqdm import tqdm
from radiance_writer import radiance_writer

ldr_path = "../test/test_baselines/CMOS/"
expand_net_path = "../test/test_baselines/ExpandNet/"
siggraph_2020_path = "../test/test_baselines/SIGGRAPH2020-rgb"
exp_brkt_path = "../test/test_baselines/ExpandNet-rgb"
gt_path = "../test/test_baselines/ground_truth/"
spad_hdr_path = "../test/SPAD-HDR/-v2.15.14-opt2/"

ldr_fnames = "{}_CMOS_monochrome.hdr"
expand_net_fnames = "{}_cmos_prediction_monochrome.hdr"
exp_brkt_2020_fnames = "{}_merged.hdr"
siggraph_2020_fnames = "img_{}.exr"
gt_fnames = "{}_gt_monochrome.hdr"
spad_hdr_fnames = "output-v2.15.14-opt2_{}.hdr"

ldr_tm_path = "../test/test_baselines_tonemapped/CMOS/"
expand_net_tm_path = "../test/test_baselines_tonemapped/ExpandNet-RGB-input"
siggraph_2020_tm_path = "../test/test_baselines_tonemapped/SIGGRAPH2020-RGB-input"
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


def main():
    pass


if __name__ == "__main__":
    main()