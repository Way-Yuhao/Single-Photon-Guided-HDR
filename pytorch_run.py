import numpy as np
import torch

"""Global Parameters"""
batch_size = 16
epoch = 1


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






if __name__ == "__main__":
    main()