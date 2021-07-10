import os
import os.path as p
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_run import tone_map, compute_l1_perc, save_hdr
from lum_fusion_model import IntensityGuidedHDRNet
from external.vgg import VGGLoss
import customDataFolder
from radiance_writer import radiance_writer

args = None
DEFAULT_CMOS_SAT = 33400 / .01


def get_args():
    parser = argparse.ArgumentParser(description="Single-Photon Camera Guided HDR Imaging")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image directory.")
    parser.add_argument("--out", "-o", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to pre-trained neural network weights.")
    parser.add_argument("--cpu", action="store_true", help="Toggle to use CPU only.")
    parser.add_argument("--saturation", "-s", type=float, help="CMOS saturation limit.")
    parser.add_argument("--gain", "-g", type=float, help="Gain applied to SPAD image; for debugging only.")
    parser.add_argument("--experimental", "-e", action="store_true", help="Toggle if using experimental data")
    parser.add_argument("--plot", "-p", action="store_true", help="Toggle to use plt.imshow().")

    args = parser.parse_args()
    return args


def set_device():
    """
    Sets device to CUDA if available
    :return: CUDA device 0, if available
    """
    if args.cpu:
        return "cpu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Testing on gpu")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on cpu")
    return device


def print_args(device):
    print("\n\n\n")
    print("\t #####################################################")
    print("\t Single-Photon Camera Guided HDR Imaging\n")
    print("\t input directory: {}".format(args.input))
    print("\t output directory: {}".format(args.out))
    print("\t weights: {}".format(args.weights))
    if device == "cpu":
        print("\t using cpu")
    else:
        print("\t CUDA is available. Testing on gpu")
    if args.saturation:
        print("\t Setting CMOS saturation limit to {}".format(args.saturation))
    if args.gain:
        print("\t Applying a gain of {} to SPAD input images".format(args.gain))
    if args.experimental:
        print("\t Using experimental data")
    if args.plot:
        print("\t Using pyplot")
    print("\t #####################################################\n\n\n")


def init_dirs():
    # handling input directory
    input_path = args.input
    cmos_path = p.join(input_path, "CMOS")
    spad_path = p.join(input_path, "SPAD")
    target_path = p.join(input_path, "ideal")
    if not(p.exists(cmos_path) and p.exists(spad_path) and (p.exists(target_path) or args.experimental)):
        raise FileNotFoundError("ERROR: input directory incomplete.")
    # handling output directory
    out_path = args.out
    if p.exists(out_path):
        print("WARNING: output path {} already exists.".format(out_path))
    else:
        os.mkdir(out_path)
    return cmos_path, spad_path, target_path


def load_hdr_data(input_path_, spad_path_, target_path_):
    """
    custom data loader that loads CMOS and SPAD inputs, and ground truth. Requires .hdr file type
    :param input_path_: path to input CMOS images
    :param spad_path_: path to input SPAD images
    :param target_path_: path to ground truth
    :return: None
    """
    data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(input_path_, spad_path_, target_path_, input_transform=None,
                                     target_transform=None, indices=None, load_all=False,
                                     monochrome=True, augment=False, cmos_sat=args.saturation),
        batch_size=1, num_workers=0, shuffle=False, sampler=None)
    return data_loader


def disp_plt(img, title="", idx=None, tone_map=False):
    """
    :param img: image to display
    :param title: title of the figure
    :param idx: index of the file, for print purposes
    :param tone_map: applies tone mapping via cv2 if set to True
    :return: None
    """
    img = img.detach().clone()

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    if img.shape[1] == 3:  # RGB
        img = img.cpu().squeeze().permute(1, 2, 0)
    else:  # monochrome
        img = img.cpu().squeeze()
        img = torch.stack((img, img, img), dim=0).permute(1, 2, 0)
    img = np.float32(img)
    if tone_map:
        tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
        img = tonemapDrago.process(img)
    plt.imshow(img)
    # compiling title
    if idx:
        title = "{} (index {})".format(title, idx)
    full_title = "{} / tone mapping={}".format(title, tone_map)
    plt.title(full_title)
    plt.show()
    return


def test(net, device):
    """
    run testing
    :param net:
    :param device:
    :param weight_path:
    :return:
    """
    net.load_state_dict(torch.load(args.weights))
    cmos_path, spad_path, target_path = init_dirs()
    test_loader = load_hdr_data(cmos_path, spad_path, target_path)
    test_iter = iter(test_loader)
    dataset_size = len(test_loader)
    print("testing on {} images".format(dataset_size))

    # experimental

    spad_gain = 1.0 if not args.gain else args.gain

    net.to(device)
    vgg_net = VGGLoss()
    vgg_net.to(device)

    loss_values = np.zeros(dataset_size)
    for i in tqdm(range(dataset_size)):
        with torch.no_grad():
            input_, spad, target = test_iter.next()
            spad *= spad_gain  # experimental
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            output = net(input_, spad)
            if not args.experimental:
                loss = compute_l1_perc(output, target, vgg_net)
                loss_values[i] = loss.item()

            if args.plot:
                disp_plt(img=input_, title="input", idx=i, tone_map=True)
                disp_plt(img=spad, title="spad", idx=i, tone_map=True)
                if not args.experimental:
                    disp_plt(img=output, title="output / loss = {:.3f}".format(loss_values[i]), idx=i, tone_map=True)
                    disp_plt(img=target, title="target", idx=i, tone_map=True)
                else:  # real data
                    disp_plt(img=output, title="output / spad gain = {}".format(spad_gain), idx=i, tone_map=True)
            save_hdr(output, p.join(args.out, "output_{}.hdr".format(i)), suppress_print=True)
    print("average loss for entire set = {}".format(loss_values.mean()))
    return


def main():
    global args
    args = get_args()
    device = set_device()
    print_args(device)
    net = IntensityGuidedHDRNet(isMonochrome=True, outputMask=False)
    test(net, device)


if __name__ == "__main__":
    main()
