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

def train_dev(net, device, tb, load_weights=False, pre_trained_params_path=None):
    """
    performs a train/dev split, and then runs training while computing dev loss at each epoch
    :param net: pytorch model object
    :param device: CUDA device, if available
    :param tb: tensorboard object
    :param load_weights: boolean flag, set true to load pre-trained weights
    :param pre_trained_params_path: path to load pre-trained network weights
    :return: None
    """
    print_params()  # print hyper parameters
    init_dir()
    net.to(device)
    net.train()

    # init vgg net
    vgg_net = VGGLoss()
    vgg_net.to(device)

    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    # splitting train/dev set
    validation_split = .2
    dataset = customDataFolder.ImageFolder(input_path, spad_path, target_path, load_all=False)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, dev_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_loader = load_hdr_data(input_path, spad_path, target_path, None, train_sampler, train_indices,
                                 num_workers_train)
    dev_loader = load_hdr_data(input_path, spad_path, target_path, None, dev_sampler, dev_indices, num_workers_val)

    print("Using cross-validation with a {:.0%}/{:.0%} train/dev split:".format(1 - validation_split, validation_split))
    print("dev set: entry {} to {} | train set: entry {} to {}"
          .format(dev_indices[0], dev_indices[-1], train_indices[0], train_indices[-1]))
    print("size of train set = {} mini-batches | size of dev set = {} mini-batches".format(len(train_loader),
                                                                                           len(dev_loader)))
    num_mini_batches = len(train_loader)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)

    # training loop
    running_train_loss = 0.0
    output, target = None, None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_, spad, target = train_iter.next()
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(input_, spad)
            loss = compute_l1_perc(output, target, vgg_net)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        # record loss values after each epoch
        cur_train_loss = running_train_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        cur_dev_loss, dev_output_sample = dev(net, device, dev_loader, ep, tb, 0, vgg_net)
        print("train loss = {:.3f} | dev loss = {:.3f} | learning rate = {:.5f}"
              .format(cur_train_loss, cur_dev_loss, scheduler.get_lr()[0]))
        running_train_loss = 0.0

        scheduler.step()

        if mini_model:
            if ep % 100 == 99:  # for every 100 epochs
                sample_train_output = output[0, :, :, :]
                disp_plt(sample_train_output, title="sample training output in epoch {}".format(ep + 1), tone_map=True)
                save_weights(net, ep)
        else:
            if ep % 20 == 19:
                sample_train_output = output[0, :, :, :]
                disp_plt(sample_train_output, title="sample training output in epoch {}".format(ep + 1), tone_map=True)
                save_weights(net, ep)

    print("finished training")
    save_weights(net, ep="{}_FINAL".format(epoch))
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
