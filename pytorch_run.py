import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from Models import AttU_Net, U_Net

"""Global Parameters"""
train_path = "../data/CMOS"
"""Hyper Parameters"""
init_lr = 0.001  # initial learning rate
batch_size = 4
epoch = 1
MAX_ITER = int(1e5)  # 1e10 in the provided file


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def print_hyper_params():
    print("######## Hyper Parameters ########")
    print("batch size = ", batch_size)
    print("epoch = ", epoch)
    return


def main():
    device = set_device()  # set device to CUDA if available
    # tb = SummaryWriter(".runs/AttU-Net")
    # net = AttU_Net(img_ch=3, output_ch=3)
    net = U_Net(img_ch=3, output_ch=3)
    net.to(device)
    # training stage
    transform = transforms.Compose([transforms.ToTensor()])  # currently without normalization
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # training stage
    for ep in range(epoch):
        for i, data in enumerate(train_loader, 0)




    # images, labels = next(iter(train_loader))
    # print("finished iter")
    # grid = torchvision.utils.make_grid(images)
    # tb.add_image("images", grid)
    # print("finished add")
    # # tb.add_graph(net, images)
    # print("finished graph")
    # # tb.close()
    # print("finished closed")

if __name__ == "__main__":
    main()
