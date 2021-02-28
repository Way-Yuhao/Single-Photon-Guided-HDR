import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from playground import customDataFolder

cmos_path = "../data/CMOS"
gt_path = "../data/ground_truth"
ds_size = 54


def imshow(img):
    npimg = img[0, :, :, :].numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    # plt.imshow(npimg)
    # plt.show()
    return npimg


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor()])

cmos_data_loader = torch.utils.data.DataLoader(
    customDataFolder.ImageFolder(cmos_path, transform=transform),
    batch_size=1, num_workers=0, shuffle=False)

gt_data_loader = torch.utils.data.DataLoader(
    customDataFolder.ImageFolder(gt_path, transform=transform),
    batch_size=1, num_workers=0, shuffle=False)


dataiter = iter(gt_data_loader)
images, labels = dataiter.next()

print(images.shape)