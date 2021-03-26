import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import warnings
from tabulate import tabulate
import matplotlib.pyplot as plt
from Models import AttU_Net, U_Net
import customDataFolder
from sequence_subset_sampler import SubsetSequenceSampler
from radiance_writer import radiance_writer

"""Global Parameters"""
version = None  # version of the model, defined in main()
train_param_path = "./model/unet/"
train_input_path = "../data/hdri_437_256x128/CMOS/"
train_label_path = "../data/hdri_437_256x128/ideal/"
down_sp_rate = 1  # down sample rate

"""Hyper Parameters"""
init_lr = 0.001  # initial learning rate
batch_size = 4
epoch = 50
MAX_ITER = int(1e5)  # 1e10 in the provided file


def set_device():
    """
    Sets device to CUDA if available
    :return: CUDA if available
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def load_hdr_data(input_path, target_path, transform=None, sampler=None):
    """
    custom dataloader that loads .hdr and .png data.
    :param input_path: path to input images
    :param target_path: path to target iamges
    :param transform: requires transform to only consist of ToTensor
    :param sampler:
    :return: dataloader object
    """
    data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(input_path, target_path, input_transform=transform, target_transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False, sampler=sampler)
    return data_loader


def load_network_weights(net, path):
    """
    loads pre-trained weights from path and prints message
    :param net:
    :param path:
    :return:
    """
    print("loading pre-trained weights from {}".format(path))
    net.load_state_dict(torch.load(path))
    return


def print_params():
    """
    prints a list of parameters to stdout
    :return: None
    """
    print("######## Hyper Parameters ########")
    print("batch size = ", batch_size)
    print("epoch = ", epoch)
    print("initial learning rate = ", init_lr)
    print("##################################")
    print("######## Other Parameters ########")
    print("down sampling rate = ", down_sp_rate)
    print("##################################")
    return


def tone_map_single(img):
    """
    Tone-mapping algorithm proposed by Nima Khademi Kalantari and Ravi Ramamoorthi.
    Deep high dynamic range imaging of dynamic scenes.
    ACM Transactions on Graphics (Proc. of ACM SIGGRAPH), 36(4):144–1, 2017.
    :param img: image tensor
    :return: tone-mapped img
    """
    mu = 5000  # amount of compression
    img = torch.log(1 + mu * img) / np.log(1 + mu)
    return img


def tone_map(output, target):
    """
    Tone-mapping algorithm proposed by Nima Khademi Kalantari and Ravi Ramamoorthi.
    Deep high dynamic range imaging of dynamic scenes.
    ACM Transactions on Graphics (Proc. of ACM SIGGRAPH), 36(4):144–1, 2017.
    :param output: output tensor of the neural network
    :param target: label tensor
    :return: tone-mapped output and target tensors
    """
    mu = 5000  # amount of compression
    output = torch.log(1 + mu * output) / np.log(1 + mu)
    target = torch.log(1 + mu * target) / np.log(1 + mu)
    return output, target


def compute_l1_loss(output, target):
    """
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
    # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, output_img)
    print("16-bit PNG save to ", path)
    return


def save_hdr(img, path):
    """
    saves 32-bit .hdr image
    :param img: image tensor of shape (1, c, h, w)
    :param path: path to save the image to
    :return: None
    """
    img = img.detach().clone()
    output_img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    radiance_writer(output_img, path)
    print("32 bit .hdr file save to ", path)
    return



def disp_plt(img, title="", tone_map=False):
    """
    :param img: image to display
    :param title: title of the figure
    :param tone_map: applies tonemapping via cv2 if set to True
    :return: None
    """
    img = img.detach().clone()
    img = img.cpu().squeeze().permute(1, 2, 0)
    img = np.float32(img)
    # img = img / img.max()  # normalize to [0, 1]
    if tone_map:
        tonemapDrago = cv2.createTonemapDrago(1.0, 1.0)
        img = tonemapDrago.process(img)
    plt.imshow(img)
    full_title = "{} / {} / tonemapping={}".format(version, title, tone_map)
    plt.title(full_title)
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
    Issue: only works when mini batch size = 1
    :param iter_:
    :param idx:
    :return:
    """
    input_data, label_data = None, None
    for _ in range(idx + 1):
        input_data, label_data = iter_.next()
    assert (input_data is not None and label_data is not None)
    return input_data, label_data


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


def dev(net, device, dev_loader, epoch_idx, tb):
    """
    performs validat
    :param net:
    :param device:
    :param input_loader:
    :param label_loader:
    :param epoch_idx:
    :param tb:
    :return: None
    """
    val_iter = iter(dev_loader)
    num_mini_batches = len(val_iter)
    net.eval()
    outputs = None
    with torch.no_grad():
        running_loss = 0.0
        for _ in range(num_mini_batches):
            input_data, label_data = val_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            running_loss += loss.item()
        # record loss values
        val_loss = running_loss / num_mini_batches
        print("val loss = {:.3f}".format(val_loss))
        tb.add_scalar('loss/dev', val_loss, epoch_idx)
    net.train()

    sample_output = outputs[1, :, :, :]
    # disp_plt(sample_output, title="sample dev output", tone_map=True)
    return val_loss, sample_output

# TODO: rename
# TODO: track model with lowest dev loss -> final model


def train_dev(net, device, tb, load_weights=False, pre_trained_params_path=None):
    print_params()  # print hyper parameters
    net.to(device)
    net.train()
    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    # splitting train/dev set
    validation_split = .2
    dataset = customDataFolder.ImageFolder(train_input_path, train_label_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, dev_indices = indices[split:], indices[:split]
    train_sampler = SubsetSequenceSampler(train_indices)
    dev_sampler = SubsetSequenceSampler(dev_indices)

    train_loader = load_hdr_data(input_path=train_input_path, target_path=train_label_path, sampler=train_sampler)
    dev_loader = load_hdr_data(input_path=train_input_path, target_path=train_label_path, sampler=dev_sampler)

    print("Using cross-validation with a {:.0%}/{:.0%} train/dev split:".format(1 - validation_split, validation_split))
    print("size of train set = {} mini-batches | size of dev set = {} mini-batches".format(len(train_loader),
                                                                                           len(dev_loader)))
    num_mini_batches = len(train_loader)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training loop
    running_loss = 0.0
    outputs, label_data = None, None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_data, label_data = train_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            optimizer.zero_grad()
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # record loss values after each epoch
        cur_val_loss, sample_val_output = dev(net, device, dev_loader, ep, tb)
        cur_train_loss = running_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        print("train loss = {:.3f} | dev loss = {:.3f}".format(cur_train_loss, cur_val_loss))
        running_loss = 0.0

        if ep % 10 == 9:  # for every 10 epochs
            sample_train_output = outputs[0, :, :, :]
            save_16bit_png(sample_train_output, path="./out_files/train_epoch_{}_{}.png".format(ep + 1, version))
            disp_plt(sample_train_output, title="sample training output in epoch {}".format(ep + 1), tone_map=True)
            save_16bit_png(sample_val_output, path="./out_files/validation_epoch_{}_{}.png".format(ep + 1, version))
            save_weights(net, ep)

    print("finished training")
    save_16bit_png(label_data[0, :, :, :], path="./out_files/sample_ground_truth.png")
    save_weights(net, ep="{}_FINAL".format(epoch))
    return


def train(net, device, tb, load_weights=False, pre_trained_params_path=None):
    print_params()  # print hyper parameters
    print("training")
    net.to(device)
    net.train()
    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    train_loader = load_hdr_data(train_input_path, train_label_path)
    num_mini_batches = len(train_loader)  # number of mini-batches per epoch
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training loop
    running_loss = 0.0
    outputs = None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_data, label_data = train_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)

            optimizer.zero_grad()
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # record loss values
        loss_cur_batch = running_loss / num_mini_batches
        print("loss = {:.3f}".format(loss_cur_batch))
        tb.add_scalar('training loss', loss_cur_batch, ep)

        if ep % 10 == 9 or True:  # for every 10 epochs
            save_16bit_png(outputs[0, :, :, :], path="./out_files/train_epoch_{}_{}.png".format(ep + 1, version))
            disp_plt(outputs[0, :, :, :], title="sample training output in epoch {}".format(ep + 1), tone_map=True)
            save_weights(net, ep)
        running_loss = 0.0

    print("finished training")
    save_16bit_png(label_data[0, :, :, :], path="./out_files/sample_ground_truth.png")
    save_weights(net, ep="{}_FINAL".format(epoch))
    return


def show_predictions(net, pre_trained_params_path):
    global batch_size
    target_idx = 9
    batch_size = 1
    print("testing on {} images".format(batch_size))
    load_network_weights(net, pre_trained_params_path)

    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    test_loader = load_hdr_data(train_input_path, train_label_path, transform)
    test_iter = iter(test_loader)

    net.eval()
    with torch.no_grad():
        input_data, label_data = select_example(test_iter, target_idx)
        outputs = net(input_data)
        loss = compute_l1_loss(outputs, label_data)

    print("loss at test time = ", loss.item())

    disp_plt(img=input_data, title="input", tone_map=True)
    disp_plt(img=outputs, title="output / loss = {:.3f}".format(loss.item()), tone_map=True)
    disp_plt(img=label_data, title="target", tone_map=True)

    save_hdr(outputs, "./out_files/test_output_{}_{}.hdr".format(version, target_idx))
    save_hdr(input_data, "./out_files/test_input_{}_{}.hdr".format(version, target_idx))
    save_hdr(label_data, "./out_files/test_ground_truth_{}_{}.hdr".format(version, target_idx))
    return


def main():
    global batch_size, version
    print("======================================================")
    version = "-v0.5.4"
    param_to_load = train_param_path + "unet{}_epoch_{}_FINAL.pth".format(version, epoch)
    tb = SummaryWriter('./runs/unet' + version)
    device = set_device()  # set device to CUDA if available
    net = U_Net(in_ch=3, out_ch=3)
    # train(net, device, tb, load_weights=False, pre_trained_params_path=param_to_load)
    # test(net, pre_trained_params_path=param_to_load)
    train_dev(net, device, tb, load_weights=False, pre_trained_params_path=param_to_load)
    tb.close()
    flush_plt()


if __name__ == "__main__":
    main()
