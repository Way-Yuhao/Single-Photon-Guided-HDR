import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from Models import AttU_Net, U_Net
from hdr_data_loader import customDataFolder

"""Global Parameters"""
train_param_path = "./model/unet/unet.pth"
train_input_path = "../data/CMOS"
train_label_path = "../data/ground_truth"
down_sp_msg_printed = False
eps = 0.000000001  # for numerical stability

"""Hyper Parameters"""
init_lr = 0.001  # initial learning rate
batch_size = 1
epoch = 1000
MAX_ITER = int(1e5)  # 1e10 in the provided file

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def load_hdr_data(path, transform):
    data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(path, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False)
    return data_loader


def print_params():
    print("######## Hyper Parameters ########")
    print("batch size = ", batch_size)
    print("epoch = ", epoch)
    print("initial learning rate = ", init_lr)
    print("################")
    return


def down_sample(input, target, down_sp_rate):
    global down_sp_msg_printed
    # m = nn.AvgPool2d(down_sp_rate, stride=down_sp_rate)
    # input = m(input)
    # target = m(target)
    input = input[:, :, ::down_sp_rate, ::down_sp_rate]
    target = target[:, :, ::down_sp_rate, ::down_sp_rate]

    return input, target


def normalize(output, target):
    # output = TF.normalize(output, mean=torch.mean(output), std=torch.std(output))
    # target = TF.normalize(target, mean=torch.mean(target), std=torch.std(target))
    target = target * 100000
    target = target / torch.max(output)
    output = output / torch.max(output)
    return output, target


def tone_map(output, target):
    mu = 5000  # amount of compression
    # mu = 1
    lb = output.min() * mu
    output = torch.log(1 + mu * output) / np.log(1 + mu)
    target = torch.log(1 + mu * target) / np.log(1 + mu)
    return output, target


def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    output, target = tone_map(output, target)
    l1_loss = criterion(output, target)
    return l1_loss


def train(net, device, tb, load_weights=False):
    print("training")
    net.train()

    if load_weights:
        net.load_state_dict(torch.load("./model/unet/unet.pth"))
        print("loading pretrained weights")

    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    train_input_loader = load_hdr_data(train_input_path, transform)
    train_label_loader = load_hdr_data(train_label_path, transform)
    assert (len(train_input_loader.dataset) == len(train_label_loader.dataset))
    num_mini_batches = len(train_input_loader)

    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training loop
    running_loss = 0.0
    for ep in range(epoch):
        print("Epoch ", ep)
        train_input_iter = iter(train_input_loader)
        train_label_iter = iter(train_label_loader)
        for _ in tqdm(range(num_mini_batches)):
            input_data, _ = train_input_iter.next()
            label_data, _ = train_label_iter.next()
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            input_data, label_data = down_sample(input_data, label_data, 4)
            input_data, label_data = normalize(input_data, label_data)
            optimizer.zero_grad()
            outputs = net(input_data)
            loss = compute_l1_loss(outputs, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print statistics
        loss_cur_batch = running_loss / batch_size
        print("loss = {:.3f}".format(loss_cur_batch))
        tb.add_scalar('training loss', loss_cur_batch, ep)
        running_loss = 0.0

    print("finished training")
    torch.save(net.state_dict(), train_param_path)
    return


def test_single(net, tb):
    global batch_size
    batch_size = 1
    print("testing")

    net.load_state_dict(torch.load("./model/unet/unet.pth"))

    test_input_path = "../data/CMOS/1/0_cmos.png"
    test_label_path = "../data/ground_truth/1/0.hdr"

    test_img = cv2.imread(test_input_path, -1).astype("float32")
    label_img = cv2.imread(test_label_path, -1).astype("float32")
    h, w, c = test_img.shape
    input_data = torch.from_numpy(test_img).view(c, h, w).unsqueeze(0)
    label_data = torch.from_numpy(label_img).view(c, h, w).unsqueeze(0)
    input_data, label_data = down_sample(input_data, label_data, 4)
    input_data, label_data = normalize(input_data, label_data)

    net.eval()
    with torch.no_grad():
        outputs = net(input_data)
        loss = compute_l1_loss(outputs, label_data)
        print("loss = ", loss)
        _, c, h, w = outputs.shape
        output_img = outputs.squeeze().permute(1, 2, 0)

        plt.imshow(output_img)
        plt.show()

        output_img = output_img.numpy()
        output_img *= 2**16
        output_img[output_img >= 2 ** 16 - 1] = 2 ** 16 - 1
        output_img = output_img.astype(np.uint16)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./sample_output.png", output_img)
        cv2.imwrite("./sample_label.png", label_img)

        tb_id = "2"
        tb.add_image("ground truth" + tb_id, label_data.squeeze())
        tb.add_image("prediction" + tb_id, outputs.squeeze())


def tb_display_test(tb):
    global batch_size
    batch_size = 1

    test_input_path = "../data/CMOS/1/0_cmos.png"
    test_label_path = "../data/ground_truth/1/0.hdr"

    # test_img = cv2.imread(test_input_path, -1).astype("float32")
    # label_img = cv2.imread(test_label_path, -1).astype("float32")
    # h, w, c = test_img.shape
    # input_data = torch.from_numpy(test_img).view(c, h, w).unsqueeze(0)
    # label_data = torch.from_numpy(label_img).view(c, h, w).unsqueeze(0)
    # input_data, label_data = down_sample(input_data, label_data, 4)
    # input_data, label_data = normalize(input_data, label_data)
    # label_img = label_data.numpy().squeeze().reshape(h, w, c)



    img = cv2.imread(test_label_path, -1)
    img = img[::4, ::4, :]
    print(img.shape)
    # brg to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(np.log(img))
    # plt.show()

    # torch.set_printoptions(precision=8)

    # img = img.astype("float32")
    # img = img / img.max()

    t = torch.from_numpy(img)

    img_rcv = t.numpy()
    plt.imshow(img_rcv)
    plt.show()

    tb.add_image("img7", t, dataformats="HWC")
    tb.flush()



def tensorboard_add_graph(tb, model):
    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    train_input_loader = load_hdr_data(train_input_path, transform)
    train_label_loader = load_hdr_data(train_label_path, transform)
    assert (len(train_input_loader.dataset) == len(train_label_loader.dataset))
    images, label = next(iter(train_input_loader))
    grid = torchvision.utils.make_grid(images)
    tb.add_images("images", grid)  # FIXME: dim mismatch
    tb.add_graph(model, images)
    return


def main():
    device = set_device()  # set device to CUDA if available
    tb = SummaryWriter('./runs/unet_cont')
    print_params()
    net = U_Net(in_ch=3, out_ch=3)
    # net.to(device)
    # train(net, device, tb, load_weights=True)
    test_single(net, tb)
    # tb_display_test(tb)
    tb.close()


if __name__ == "__main__":
    main()
