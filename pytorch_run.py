import os
import os.path as p
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from lum_fusion_model import IntensityGuidedHDRNet
import customDataFolder
from radiance_writer import radiance_writer
from external.vgg import VGGLoss
from ablation_study.model_no_attention import HDRNetNoAttention
from ablation_study.model_no_spad import HDRNetNoSpad

"""Global Parameters"""
version = None  # version of the model, defined in main()
monochrome = True  # True if in monochromatic mode
mini_model = False  # True if only run train/dev on the small subset of the full dataset
redirect_plt = not mini_model  # True if saving pyplot into PNG files, False if displaying through plt.show()
visualize_mask = False  # True if visualizing the 2nd attention mask; for debugging only
down_sp_rate = 1  # down sample rate; OBSOLETE
train_param_path = "./model/unet/"  # path for loading/saving network weights; need to specify version in main()
plt_path = "./plt/"  # path for saving pyplot outputs when redirect_plt == True

"""train & dev set"""
if mini_model:
    input_path = "../data/small_shuffled/CMOS/"
    target_path = "../data/small_shuffled/ideal/"
    spad_path = "../data/small_shuffled/SPAD/"
else:  # full model
    input_path = "../data/combined_shuffled/CMOS/"
    target_path = "../data/combined_shuffled/ideal/"
    spad_path = "../data/combined_shuffled/SPAD/"

"""test set"""
# input_path = "../data/test/CMOS/"
# target_path = "../data/test/ideal/"
# spad_path = "../data/test/SPAD/"

"""real data"""
# input_path = "../data/real_data/fire2/CMOS/"
# target_path = "../data/real_data/fire2/ideal/"
# spad_path = "../data/real_data/fire2/SPAD/"

"""Hyper Parameters"""
init_lr = 0.001      # initial learning rate
epoch = 2000         # number of epochs used in training

if mini_model:
    num_workers_train = 0
    num_workers_val = 0
    batch_size = 16
else:  # full model
    num_workers_train = 16
    num_workers_val = 8
    batch_size = 16

"""Simulation Parameters"""
CMOS_fwc = 33400  # full well capacity of the CMOS sensor
CMOS_T = .01  # exposure time of the CMOS sensor, in seconds
CMOS_sat = CMOS_fwc / CMOS_T  # saturation value of the CMOS simulated images


def set_device():
    """
    Sets device to CUDA if available
    :return: CUDA device 0, if available
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def load_hdr_data(input_path_, spad_path_, target_path_, transform=None, sampler=None, indices=None,
                  _num_workers=0, load_all=True, augment=True):
    """
    custom data loader that loads CMOS and SPAD inputs, and ground truth. Requires .hdr file type
    :param input_path_: path to input CMOS images
    :param spad_path_: path to input SPAD images
    :param target_path_: path to ground truth
    :param transform: transform applied to loaded tensors; None by default
    :param sampler: sampler used to separate train / dev sets
    :param indices: OBSOLETE
    :param _num_workers: number of workers for pytorch data loader
    :param load_all: set True if loading all images into RAM; requires high RAM space
    :param augment: set True if applying data augmentation
    :return: None
    """
    data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(input_path_, spad_path_, target_path_, input_transform=transform,
                                     target_transform=transform, indices=indices, load_all=load_all,
                                     monochrome=monochrome, augment=augment, cmos_sat=CMOS_sat),
        batch_size=batch_size, num_workers=_num_workers, shuffle=False, sampler=sampler)
    return data_loader


def load_network_weights(net, path):
    """
    loads pre-trained weights from path and prints message
    :param net: pytorch network object
    :param path: path to load network weights
    :return: None
    """
    print("loading pre-trained weights from {}".format(path))
    net.load_state_dict(torch.load(path))
    return


def print_params():
    """
    prints a list of parameters to stdout
    :return: None
    """
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("redirecting plt outputs = {}".format(redirect_plt))
    print("Monochrome = {}".format(monochrome))
    if visualize_mask:
        print("WARNING: visualizing mask")
    print("######## Hyper Parameters ########")
    print("batch size = ", batch_size)
    print("epoch = ", epoch)
    print("initial learning rate = ", init_lr)
    print("##################################")
    print("######## Other Parameters ########")
    print("down sampling rate = ", down_sp_rate)
    print("##################################")
    if mini_model:
        print("working with MINI model")
    else:
        print("working with FULL model")
    return


def init_dir():
    """
    initiate a new directory for plt outputs of the current version
    :return:
    """
    global plt_path
    plt_dir = os.path.join(plt_path, version)
    if os.path.exists(plt_dir):
        # raise FileExistsError("ERROR: directory {} for pyplot outputs already exists. Please remove.".format(plt_dir))
        pass
    else:
        os.mkdir(plt_dir)
    return


def tone_map(output, target):
    """
    Tone-mapping algorithm proposed by Nima Khademi Kalantari and Ravi Ramamoorthi.
    Deep high dynamic range imaging of dynamic scenes.
    ACM Transactions on Graphics (Proc. of ACM SIGGRAPH), 36(4):144–1, 2017.
    :param output: output tensor of the neural network
    :param target: ground truth tensor
    :return: tone-mapped output and ground truth tensors
    """
    mu = 2000  # amount of compression
    output = torch.log(1 + mu * output) / np.log(1 + mu)
    target = torch.log(1 + mu * target) / np.log(1 + mu)
    return output, target


def compute_l1_perc(output, target, vgg_net):
    """
    applies tone mapping and computes a linear combination of L1 and perceptual loss
    :param vgg_net: pre-trained vgg network used to compute perceptual loss
    :param output: output tensor of the neural network
    :param target: ground truth tensor
    :return: a linear combination of tone mapped L1 loss and tone mapped perceptual loss
    """
    if visualize_mask:
        return None
    elif monochrome:
        output = torch.cat((output, output, output), dim=1)

    l1_criterion = nn.L1Loss()
    output, target = tone_map(output, target)
    l1_loss = l1_criterion(output, target)

    with torch.no_grad():
        perc_loss = vgg_net(output, target)

    total_loss = l1_loss + .1 * perc_loss
    return total_loss


def compute_l1_loss(output, target):
    """
    OBSOLETE
    computes the L1 loss between output and target after tone-mapping
    :param output: output tensor of the neural network
    :param target: label tensor
    :return: L1 loss
    """
    criterion = nn.L1Loss()
    output, target = tone_map(output, target)
    l1_loss = criterion(output, target)
    return l1_loss


def save_16bit_png(img, path):
    """
    saves 16-bit PNG image via cv2
    :param img: image tensor of shape (1, c, h, w)
    :param path: path to save the image to
    :return: None
    """
    img = img.detach().clone()
    output_img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    output_img *= 2 ** 16
    output_img[output_img >= 2 ** 16 - 1] = 2 ** 16 - 1
    output_img = output_img.astype(np.uint16)
    cv2.imwrite(path, output_img)
    print("16-bit PNG save to ", path)
    return


def save_hdr(img, path, suppress_print=False):
    """
    saves 32-bit .hdr image
    :param img: image tensor of shape (1, c, h, w)
    :param path: path to save the image to
    :param suppress_print: True if suppressing stdout
    :return: None
    """
    img = img.detach().clone()
    if img.shape[1] == 1:
        img = torch.stack((img, img, img), dim=1).squeeze(dim=2)
    output_img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    radiance_writer(output_img, path)
    if not suppress_print:
        print("32 bit .hdr file save to ", path)
    return


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
    full_title = "{} / {} / tone mapping={}".format(version, title, tone_map)
    plt.title(full_title)

    if redirect_plt:
        fname = "{}_{}.png".format(version, title)
        plt.savefig(os.path.join(plt_path, version, fname))
    else:
        plt.show()
    return


def flush_plt():
    """
    flushes the plot in SciView with a blank figure. This function is only useful in PyCharm environment
    :return: None
    """
    blank_img = np.ones((128, 256, 3))
    plt.imshow(blank_img)
    plt.title("blank image")
    plt.show()
    return


def select_example(iter_, idx):
    """
    iterates through a dataset and returns a tuple of (input, spad, target) according to a specified index
    WARNING: only works when mini batch size = 1
    :param iter_: iterator of data loader
    :param idx: index of the desired tuple (input, spad, target)
    :return:
    """
    input_, spad, target = None, None, None
    for _ in range(idx + 1):
        input_, spad, target = iter_.next()
    assert (input_ is not None and spad is not None and target is not None)
    return input_, spad, target


def save_weights(net, ep=None):
    """
    saves weights of the neural network
    :param ep: number of epochs trained
    :param net: torch network object
    :return: None
    """
    if epoch is None:
        filename = train_param_path + "unet{}.pth".format(version)
    else:
        filename = train_param_path + "unet{}_epoch_{}.pth".format(version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def disp_sample(input_, spad, output, target, idx=0, msg=""):
    """
    helper function that displays a triplet of input, output, and target
    :param input_: CMOS input tensor
    :param spad: SPAD input tensor
    :param output: network output tensor
    :param target: ground truth tensor
    :param idx: index in the data loader, for title
    :param msg: message to be printed
    :return: None
    """
    if len(input_.size()) == 4:
        if input_ is not None:
            disp_plt(input_[idx, :, :, :], title=msg + " / input", tone_map=True)
        if spad is not None:
            disp_plt(spad[idx, :, :, :], title=msg + " / spad", tone_map=True)
        if output is not None:
            disp_plt(output[idx, :, :, :], title=msg + " / outputs", tone_map=True)
        if target is not None:
            disp_plt(target[idx, :, :, :], title=msg + " / target", tone_map=True)
    elif len(input_.size()) == 3:
        if input_ is not None:
            disp_plt(input_, title=msg + " / input", tone_map=True)
        if spad is not None:
            disp_plt(spad, title=msg + " / spad", tone_map=True)
        if output is not None:
            disp_plt(output, title=msg + " / outputs", tone_map=True)
        if target is not None:
            disp_plt(target, title=msg + " / target", tone_map=True)
    else:
        raise Exception("ERROR: number of dimension of image tensor is neither 3 nor 4")

    flush_plt()
    return


def median_filter(img):
    """
    OBSOLETE
    applies median filter to an image tensor
    :param img: numpy image array of shape (h, w, 3)
    :return: median filtered image tensor
    """
    return cv2.medianBlur(img, 3)


def dev(net, device, dev_loader, epoch_idx, tb, target_idx=0, vgg_net=None):
    """
    computes the loss on the dev set, without back propagation
    :param net: pytorch model object
    :param device: CUDA device, if available
    :param dev_loader: dev set data loader
    :param epoch_idx: current epoch number, for printing
    :param tb: tensorboard object
    :param target_idx: target sample index to return
    :param vgg_net: pre-trained vgg network used to compute perceptual loss
    :return: loss for entire dev set (1 epoch), sample output in dev set
    """
    dev_iter = iter(dev_loader)
    num_mini_batches = len(dev_iter)
    # net.eval()
    output = None
    with torch.no_grad():
        running_loss = 0.0
        for _ in range(num_mini_batches):
            input_, spad, target = dev_iter.next()
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            output = net(input_, spad)
            loss = compute_l1_perc(output, target, vgg_net)
            running_loss += loss.item()
        # record loss values
        dev_loss = running_loss / num_mini_batches
        tb.add_scalar('loss/dev', dev_loss, epoch_idx)
    sample_output = output[target_idx, :, :, :]

    return dev_loss, sample_output


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
    dataset = customDataFolder.ImageFolder(input_path, spad_path, target_path, load_all=False, cmos_sat=CMOS_sat)
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



def show_pred_all(net, device, loader_iter, size):
    """
    displaying network output for all inputs in the dataset
    :param net: pytorch object
    :param device: CUDA device
    :param loader_iter: iterator of the data loader
    :param size: size of the data set
    :return: None
    """
    print("displaying network output for all inputs in the dataset; size = {}".format(size))

    if os.path.exists("./out_files/pred_all/{}/".format(version)):
        raise Exception("Error: ./out_files/pred_all/{}/ already exists".format(version))
    else:
        os.mkdir("./out_files/pred_all/{}/".format(version))
        print("storing outputs in new directory ./out_files/pred_all/{}/".format(version))
    loss_values = np.zeros(size)
    for i in tqdm(range(size)):
        with torch.no_grad():
            input_, spad, target = loader_iter.next()
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            output = net(input_, spad)
            loss = compute_l1_loss(output, target)
            loss_values[i] = loss.item()
            save_hdr(output, "./out_files/pred_all/{}/output{}_{}.hdr".format(version, version, i))
    print("average loss for entire set = {}".format(loss_values.mean()))
    return


def show_predictions(net, device, target_idx, pre_trained_params_path):
    """
    displays and saves a select sample output
    :param net: pytorch object
    :param device: CUDA device
    :param target_idx: index in the dataset to be run
    :param pre_trained_params_path: path to load pre-trained weights
    :return: None
    """
    global batch_size, redirect_plt
    redirect_plt = False
    batch_size = 1
    load_network_weights(net, pre_trained_params_path)
    test_loader = load_hdr_data(input_path, spad_path, target_path, load_all=False, augment=False)
    test_iter = iter(test_loader)

    net.to(device)
    vgg_net = VGGLoss()
    vgg_net.to(device)

    if target_idx is -1:  # batch
        show_pred_all(net, device, test_iter, len(test_loader))
    else:  # single
        print("testing on {} images, index = {}".format(batch_size, target_idx))
        # net.eval()
        with torch.no_grad():
            input_, spad, target = select_example(test_iter, target_idx)
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            output = net(input_, spad)
            loss = compute_l1_perc(output, target, vgg_net)

        # handling the case where no loss value is returned (e.g. when visualizing mask)
        loss_value = 0.0
        if loss is not None:
            loss_value = loss.item()
        print("loss at test time = ", loss_value)

        disp_plt(img=input_, title="input", idx=target_idx, tone_map=True)
        disp_plt(img=spad, title="spad", idx=target_idx, tone_map=True)
        disp_plt(img=output, title="output / loss = {:.3f}".format(loss_value), idx=target_idx, tone_map=True)
        disp_plt(img=target, title="target", idx=target_idx, tone_map=True)

        save_hdr(output, "./out_files/test_output{}_{}.hdr".format(version, target_idx))
        save_hdr(input_, "./out_files/test_input{}_{}.hdr".format(version, target_idx))
        save_hdr(target, "./out_files/test_ground_truth{}_{}.hdr".format(version, target_idx))
    return


def show_prediction_real_data(net, device, pre_trained_params_path):
    """
    displays network prediction on real data.
    Requirements:
        * specify CMOS saturation limit in CMOS_sat
        * may need to manually adjust normalization in data loading stage
    :param net: pytorch object
    :param device: CUDA device
    :param pre_trained_params_path: path to load pre-trained weights
    :return: None
    """
    global redirect_plt, batch_size

    print("Showing prediction on real data.\nCMOS saturation limit set to {}".format(CMOS_sat))
    redirect_plt = False
    batch_size = 1

    load_network_weights(net, pre_trained_params_path)
    test_loader = load_hdr_data(input_path, spad_path, target_path, load_all=False, augment=False)
    test_iter = iter(test_loader)
    size = len(test_loader)

    net.to(device)
    vgg_net = VGGLoss()
    vgg_net.to(device)

    if os.path.exists("./out_files/pred_real_data/{}/".format(version)):
        # raise Exception("Error: ./out_files/pred_real_data/{}/ already exists".format(version))
        pass
    else:
        os.mkdir("./out_files/pred_real_data/{}/".format(version))
        print("storing outputs in new directory ./out_files/pred_real_data/{}/".format(version))

    for i in tqdm(range(size)):
        with torch.no_grad():
            input_, spad, target = test_iter.next()
            input_, spad, target = input_.to(device), spad.to(device), target.to(device)
            output = net(input_, spad)

            disp_plt(img=input_, title="input", idx=i, tone_map=True)
            disp_plt(img=spad, title="spad", idx=i, tone_map=True)
            disp_plt(img=output, title="output", idx=i, tone_map=True)
            save_hdr(output, "./out_files/pred_real_data/{}/output{}_{}.hdr".format(version, version, i))
    return


def main():
    """
    main function of the script. Select among the following options:
        * train_dev(): performs training while computing dev loss
        * show_predictions(): performs testing across dataset or on a specified index in the dataset
        * show_prediction_real_data(): performs testing on real data
    :return: None
    """
    global batch_size, version
    print("======================================================")
    # define version of the network here; used in tensorboard, loading/saving network weights
    version = "-v3.2.0"
    param_to_load = None
    # param_to_load = p.join(train_param_path, "unet{}_epoch_{}_FINAL.pth".format(version, epoch))
    # param_to_load = p.join(train_param_path, "unet-v2.15.14_epoch_1819_OPT.pth")
    tb = SummaryWriter('./runs/unet' + version)
    device = set_device()  # set device to CUDA if available
    net = IntensityGuidedHDRNet(isMonochrome=monochrome, outputMask=visualize_mask) # for output mask, change true, and remove train_dev
    train_dev(net, device, tb, load_weights=False, pre_trained_params_path=param_to_load)
    #show_predictions(net, device, target_idx=-1, pre_trained_params_path=param_to_load)
    #show_prediction_real_data(net, device, pre_trained_params_path=param_to_load)

    ########## ablation study ############
    # net_no_att = HDRNetNoAttention(isMonochrome=True)
    # train_dev(net_no_att, device, tb, load_weights=False, pre_trained_params_path=None)

    # net_no_spad = HDRNetNoSpad(isMonochrome=True)
    # train_dev(net_no_spad, device, tb, load_weights=False, pre_trained_params_path=None)



    tb.close()  # closes tensorbaord
    flush_plt()  # useful only in PyCharm


if __name__ == "__main__":
    main()
