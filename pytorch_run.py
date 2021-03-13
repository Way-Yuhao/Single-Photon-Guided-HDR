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
from tabulate import tabulate
import warnings
import matplotlib.pyplot as plt
from Models import AttU_Net, U_Net
import customDataFolder
from sequence_subset_sampler import SubsetSequenceSampler

"""Global Parameters"""
version = None  # version of the model, defined in main()
train_param_path = "./model/unet/unet{}.pth"
train_input_path = "../data/hdri_437_256x128/CMOS"
train_label_path = "../data/hdri_437_256x128/ideal"
down_sp_rate = 1  # down sample rate

"""Hyper Parameters"""
init_lr = 0.001  # initial learning rate
batch_size = 4
epoch = 500
MAX_ITER = int(1e5)  # 1e10 in the provided file


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def load_hdr_data(path, transform, sampler=None):
    data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(path, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False, sampler=sampler)
    return data_loader


def print_params():
    print("######## Hyper Parameters ########")
    print("batch size = ", batch_size)
    print("epoch = ", epoch)
    print("initial learning rate = ", init_lr)
    print("##################################")
    print("######## Other Parameters ########")
    print("down sampling rate = ", down_sp_rate)
    print("##################################")
    return


def down_sample(input, target, down_sp_rate):
    if down_sp_rate is 1:
        return input, target
    input = input[:, :, ::down_sp_rate, ::down_sp_rate]
    target = target[:, :, ::down_sp_rate, ::down_sp_rate]
    return input, target


def normalize(input, target):
    target = target
    target = target / 2 ** 16
    input = input / 2 ** 16
    return input, target


def tone_map_single(img):
    mu = 5000  # amount of compression
    img = torch.log(1 + mu * img) / np.log(1 + mu)
    return img


def tone_map(output, target):
    mu = 5000  # amount of compression
    output = torch.log(1 + mu * output) / np.log(1 + mu)
    target = torch.log(1 + mu * target) / np.log(1 + mu)
    return output, target


def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    output, target = tone_map(output, target)
    l1_loss = criterion(output, target)
    return l1_loss


def save_16bit_png(img, path):
    img = img.detach().clone()
    output_img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    output_img *= 2 ** 16
    output_img[output_img >= 2 ** 16 - 1] = 2 ** 16 - 1
    output_img = output_img.astype(np.uint16)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, output_img)
    print("16-bit PNG save to ", path)
    return


def disp_plt(img, title="", normalize=False):
    """
    :param img: image to display
    :param title: title of the figure
    :param path: path to save the figure. If empty or None, this function will not save the figure
    :param normalize: set to True if intend to normalize the image to [0, 1]
    :return: None
    """
    img = img.detach().clone()
    img = img / img.max() if normalize else img
    plt.imshow(img.cpu().squeeze().permute(1, 2, 0))
    plt.title(title)
    plt.show()
    return


def flush_plt():
    """
    flushes the plot in SciView with a blank figure. This function is only useful in PyCharm environment
    :return:
    """
    blank_img = np.ones((128, 256, 3))
    plt.imshow(blank_img)
    plt.title("blank image")
    plt.show()
    return


def select_target_example(batch_idx, eg_idx, input_iter, label_iter, mode=None, batch_size=None):
    """
    Issue: only works when mini batch size = 1
    :param batch_idx:
    :param eg_idx:
    :param input_iter:
    :param label_iter:
    :param mode:
    :param batch_size:
    :return:
    """
    input_data, label_data = None, None
    for _ in range(batch_idx + 1):
        input_data, _ = input_iter.next()
        label_data, _ = label_iter.next()
    assert (input_data is not None and label_data is not None)
    # disp_plt(input_data, "input: {}th example in {}th mini-batch. {}ing with batch size = {}".format(batch_idx, eg_idx, mode, batch_size), True)
    # disp_plt(label_data, "label: {}th example in {}th mini-batch. {}ing with batch size = {}".format(batch_idx, eg_idx, mode, batch_size), False)
    return input_data, label_data


def cross_validation_test(net, device, input_loader, label_loader, epoch_idx, tb):
    val_input_iter = iter(input_loader)
    val_label_iter = iter(label_loader)
    num_mini_batches = len(input_loader)
    net.eval()
    outputs = None
    with torch.no_grad():
        running_loss = 0.0
        for _ in range(num_mini_batches):
            input_data, _ = val_input_iter.next()
            label_data, _ = val_label_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            input_data, label_data = down_sample(input_data, label_data, down_sp_rate)
            input_data, label_data = normalize(input_data, label_data)
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            running_loss += loss.item()
        # record loss values
        val_loss = running_loss / num_mini_batches
        print("val loss = {:.3f}".format(val_loss))
        tb.add_scalar('loss/dev', val_loss, epoch_idx)
    net.train()

    sample_output = outputs[0, :, :, :]
    return val_loss, sample_output


def cross_validation(net, device, tb, load_weights=False):
    print_params()  # print hyper parameters
    net.to(device)
    net.train()
    if load_weights:
        net.load_state_dict(torch.load(train_param_path))
        print("loading pre-trained weights")

    # splitting train/dev set
    validation_split = .2
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_input = customDataFolder.ImageFolder(train_input_path, transform=transform)
    dataset_label = customDataFolder.ImageFolder(train_label_path, transform=transform)
    assert (len(dataset_input) == len(dataset_label))
    dataset_size = len(dataset_input)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_input_indices, val_input_indices = indices[split:], indices[:split]
    train_label_indices, val_label_indices = indices[split:], indices[:split]
    train_input_sampler = SubsetSequenceSampler(train_input_indices)
    valid_input_sampler = SubsetSequenceSampler(val_input_indices)
    train_label_sampler = SubsetSequenceSampler(train_label_indices)
    valid_label_sampler = SubsetSequenceSampler(val_label_indices)

    train_input_loader = load_hdr_data(path=train_input_path, transform=transform, sampler=train_input_sampler)
    train_label_loader = load_hdr_data(path=train_label_path, transform=transform, sampler=train_label_sampler)
    valid_input_loader = load_hdr_data(path=train_input_path, transform=transform, sampler=valid_input_sampler)
    valid_label_loader = load_hdr_data(path=train_label_path, transform=transform, sampler=valid_label_sampler)
    print("Using cross-validation with a {:.0%}/{:.0%} train/dev split:".format(1 - validation_split, validation_split))
    print("size of train set = {} mini-batches | size of dev set = {} mini-batches".format(len(train_input_loader),
                                                                                           len(valid_input_loader)))
    num_mini_batches = len(train_input_loader)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training loop
    running_loss = 0.0
    outputs = None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_input_iter = iter(train_input_loader)
        train_label_iter = iter(train_label_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_data, _ = train_input_iter.next()
            label_data, _ = train_label_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            # input_data, label_data = down_sample(input_data, label_data, down_sp_rate)
            input_data, label_data = normalize(input_data, label_data)
            optimizer.zero_grad()
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # record loss values after each epoch
        cur_val_loss, sample_val_output = cross_validation_test(net, device, valid_input_loader, valid_label_loader, ep, tb)
        cur_train_loss = running_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        print("train loss = {:.3f} | valid loss = {:.3f}".format(cur_train_loss, cur_val_loss))
        running_loss = 0.0

        if ep % 10 == 9:  # for every 10 epochs
            sample_train_output = outputs[0, :, :, :]
            save_16bit_png(sample_train_output, path="./out_files/train_epoch_{}_{}.png".format(ep + 1, version))
            disp_plt(sample_train_output, title="sample training output in epoch {} // Model version {}".format(ep + 1, version))

            save_16bit_png(sample_val_output, path="./out_files/validation_epoch_{}_{}.png".format(ep + 1, version))

    print("finished training")
    save_16bit_png(label_data[0, :, :, :], path="./out_files/sample_ground_truth.png")
    # tb.add_image("train_final_output/linear", outputs.detach().cpu().squeeze())
    # tb.add_image("train_final_output/tonemapped", tone_map_single(outputs.detach().cpu().squeeze()))
    # tb.add_image("train_final_output/normalized", outputs.detach().cpu().squeeze() / outputs.max())

    torch.save(net.state_dict(), train_param_path.format(version))
    return


def train(net, device, tb, load_weights=False):
    print_params()  # print hyper parameters
    print("training")
    net.to(device)
    net.train()
    if load_weights:
        net.load_state_dict(torch.load(train_param_path))
        print("loading pre-trained weights")
    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    train_input_loader = load_hdr_data(train_input_path, transform)
    train_label_loader = load_hdr_data(train_label_path, transform)
    assert (len(train_input_loader.dataset) == len(train_label_loader.dataset))
    num_mini_batches = len(train_input_loader)  # number of mini-batches per epoch

    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training loop
    running_loss = 0.0
    outputs = None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_input_iter = iter(train_input_loader)
        train_label_iter = iter(train_label_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_data, _ = train_input_iter.next()
            label_data, _ = train_label_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            input_data, label_data = down_sample(input_data, label_data, down_sp_rate)
            input_data, label_data = normalize(input_data, label_data)
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

        # if ep % 100 == 99:  # for every 100 epochs
        if True:
            save_16bit_png(outputs[0, :, :, :], path="./out_files/train_epoch_{}_version_{}.png".format(ep + 1, version))
            disp_plt(outputs[0, :, :, :],
                     title="sample training output in epoch {} // Model version {}".format(ep + 1, version))
        running_loss = 0.0

    save_16bit_png(label_data[0, :, :, :], path="./out_files/sample_ground_truth.png")

    print("finished training")
    # tb.add_image("train_final_output/linear", outputs.detach().cpu().squeeze())
    # tb.add_image("train_final_output/tonemapped", tone_map_single(outputs.detach().cpu().squeeze()))
    # tb.add_image("train_final_output/normalized", outputs.detach().cpu().squeeze() / outputs.max())

    torch.save(net.state_dict(), train_param_path.format(version))
    return


def test(net, tb):
    global batch_size
    target_batch_idx = 1
    target_eg_idx = 0
    batch_size = 1
    print("testing on {} images".format(batch_size))
    net.load_state_dict(torch.load(train_param_path.format(version)))

    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    test_input_loader = load_hdr_data(train_input_path, transform)
    test_label_loader = load_hdr_data(train_label_path, transform)
    assert (len(test_input_loader.dataset) == len(test_label_loader.dataset))

    test_input_iter = iter(test_input_loader)
    test_label_iter = iter(test_label_loader)

    net.eval()
    with torch.no_grad():
        input_data, label_data = \
            select_target_example(target_batch_idx, target_eg_idx,
                                  test_input_iter, test_label_iter, mode="test", batch_size=batch_size)
        input_data, label_data = down_sample(input_data, label_data, down_sp_rate)
        input_data, label_data = normalize(input_data, label_data)
        outputs = net(input_data)
        loss = compute_l1_loss(outputs, label_data)

    print("loss at test time = ", loss.item())
    tb.add_image("test_output/linear", outputs.detach().cpu().squeeze())
    tb.add_image("test_output/tonemapped", tone_map_single(outputs.detach().cpu().squeeze()))
    tb.add_image("test_output/normalized", outputs.detach().cpu().squeeze() / outputs.max())

    disp_plt(img=input_data, title="model version {}/ input".format(version), normalize=False)
    disp_plt(img=outputs, title="model version {}/ test output".format(version), normalize=False)
    disp_plt(img=label_data, title="model version {}/ ground truth".format(version), normalize=False)

    save_16bit_png(outputs, "./out_files/test_output_{}_{}.png".format(version, target_batch_idx))
    save_16bit_png(input_data, "./out_files/test_input_{}_{}.png".format(version, target_batch_idx))
    save_16bit_png(label_data, "./out_files/test_ground_truth_{}_{}.png".format(version, target_batch_idx))
    return


def main():
    global batch_size, version
    version = "-v0.4.6-test"
    tb = SummaryWriter('./runs/unet' + version)
    device = set_device()  # set device to CUDA if available
    net = U_Net(in_ch=3, out_ch=3)
    # train(net, device, tb, load_weights=False)
    # test(net, tb)
    cross_validation(net, device, tb)
    tb.close()
    # flush_plt()


if __name__ == "__main__":
    main()
