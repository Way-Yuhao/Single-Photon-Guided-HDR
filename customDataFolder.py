from torchvision.datasets.vision import VisionDataset
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import os
import os.path as p
import numpy as np
from natsort import natsorted
# from pytorch_run import disp_plt, disp_sample


def cv_loader(path):
    """
    loads .hdr file via cv2, then converts color to rgb
    :param path: path to image file
    :return: img ndarray
    """
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    return img


def down_sample(input_, target, down_sp_rate):
    """
    OBSOLETE
    down-samples input and label at a given down sampling rate
    :param input_: input tensor of shape (m, c, h, w)
    :param target: label tensor of shape (m, c, h, w)
    :param down_sp_rate: a positive integer specifying the down sampling rate
    :return: down-sampled input, label pair
    """
    if down_sp_rate is 1:
        return input_, target
    input_ = input_[:, :, ::down_sp_rate, ::down_sp_rate]
    target = target[:, :, ::down_sp_rate, ::down_sp_rate]
    return input_, target


def cvt_monochrome(input_, spad, target):
    """
    converts BGR color image to monochrome. The output contains 3 copies of green channel
    :param input_: CMOS input
    :param spad: SPAD input
    :param target: ground truth
    :return: monochrome images
    """
    input_ = input_[1, :, :]
    input_ = torch.stack((input_, input_, input_), dim=0)

    spad = spad[1, :, :]
    spad = torch.stack((spad, spad, spad), dim=0)

    target = target[1, :, :]
    target = torch.stack((target, target, target), dim=0)

    return input_, spad, target


class ImageFolder(VisionDataset):
    """A generic data loader where the images are arranged in this way:

        input_dir/0.png
        input_dir/1.png
        input_dir/2.png

        spad_dir/0.hdr
        spad_dir/1.hdr
        spad_dir/2.hdr

        target_dir/0.hdr
        target_dir/1.hdr
        target_dir/2.hdr

    Args:
        input_dir (string): input image directory path.
        target_dir (string): target image directory path.
        input_transform (callable, optional): A function/transform that takes in an cv2 image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in an cv2 image
            and returns a transformed version.
    """

    def __init__(self, input_dir, spad_dir, target_dir, input_transform=None, spad_transform=None,
                 target_transform=None, indices=None, load_all=True, monochrome=False, augment=True,
                 cmos_sat=None):

        self.load_all = load_all
        self.inputs = natsorted(os.listdir(input_dir))
        self.targets = natsorted(os.listdir(target_dir))
        self.spad_inputs = natsorted(os.listdir(spad_dir))
        self.input_dir = input_dir
        self.spad_dir = spad_dir
        self.target_dir = target_dir
        self.isMonochrome = monochrome
        self.augment = augment
        if cmos_sat is not None:
            self.cmos_saturation = cmos_sat
        else:
            raise ValueError("ERROR: undefined CMOS saturation")

        if input_transform is not None:
            self.input_transform = input_transform
        else:
            self.input_transform = transforms.Compose([transforms.ToTensor()])
        if spad_transform is not None:
            self.spad_transform = spad_transform
        else:
            self.spad_transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.check_files()

        if self.load_all:
            self.dataset = []
            self.indices = indices
            entries = natsorted(os.listdir(input_dir))
            for i in tqdm(range(len(entries))):
                if i in self.indices:
                    input_sample = cv_loader(p.join(self.input_dir, self.inputs[i]))
                    spad_sample = cv_loader(p.join(self.spad_dir, self.spad_inputs[i]))
                    target_sample = cv_loader(p.join(self.target_dir, self.targets[i]))
                    self.dataset.append([input_sample, spad_sample, target_sample])
                else:
                    self.dataset.append(None)
            print("successfully loaded dataset to memory")
        return

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        if not self.load_all:
            input_sample = cv_loader(p.join(self.input_dir, self.inputs[item]))
            spad_sample = cv_loader(p.join(self.spad_dir, self.spad_inputs[item]))
            target_sample = cv_loader(p.join(self.target_dir, self.targets[item]))
        else:
            input_sample = self.dataset[item][0]
            spad_sample = self.dataset[item][1]
            target_sample = self.dataset[item][2]

        input_sample = self.input_transform(input_sample)
        spad_sample = self.spad_transform(spad_sample)
        target_sample = self.target_transform(target_sample)
        input_sample, spad_sample, target_sample = self.data_augmentation(input_sample, spad_sample, target_sample)
        input_sample, spad_sample, target_sample = self.normalize(input_sample, spad_sample, target_sample)
        if self.isMonochrome:
            input_sample, spad_sample, target_sample = cvt_monochrome(input_sample, spad_sample, target_sample)

        return input_sample, spad_sample, target_sample

    def normalize(self, input_, spad, target):
        """
        normalizes input to [0, 255], spad and target to [0, >255]
        :param input_: input tensor. Expects original input image files to be 16-bit PNG (uint16)
        :param spad: side input tensor. Expects original input image files to be 32-bit .hdr (float32)
        :param target: target tensor. Expects original label image files to be 32-bit .hdr (float32)
        :return: normalized input and label
        """
        input_ = input_ / self.cmos_saturation * 255
        spad = spad / self.cmos_saturation * 255
        target = target / self.cmos_saturation * 255

        return input_, spad, target

    def data_augmentation(self, input_, spad, target):
        """
        applies a sequence of data augmentations
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :return: augmented inputs and ground truth
        """
        if self.augment is False:
            pass
        else:
            input_, spad, target = self.random_crop(input_, spad, target)
            input_, spad, target = self.random_horizontal_flip(input_, spad, target)
            input_, spad, target = self.random_rotation(input_, spad, target)
        return input_, spad, target

    def random_crop(self, input_, spad, target):
        """
        applies random cropping and returns a section of CMOS input, SPAD input, and ground truth
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :return: 3 crops of size defined in function
        """
        crop_width = 512
        crop_height = 256
        diff = 4
        sat_bound = .1 * crop_width * crop_height

        max_h = input_.shape[1] - crop_height
        max_w = input_.shape[2] - crop_width

        is_saturated = False
        input_crop, spad_crop, target_crop = None, None, None
        counter = 10
        while not is_saturated and counter > 0:
            h = np.random.randint(0, max_h / 2) * 2  # random even numbers
            w = np.random.randint(0, max_w / 2) * 2
            input_crop = input_[:, h: h + crop_height, w: w + crop_width]
            target_crop = target[:, h: h + crop_height, w: w + crop_width]
            spad_crop = spad[:, int(h / diff): int((h + crop_height) / diff),
                        int(w / diff): int((w + crop_width) / diff)]
            if torch.sum(input_crop > self.cmos_saturation) > sat_bound:
                is_saturated = True
            else:
                counter -= 1

        return input_crop, spad_crop, target_crop

    def random_horizontal_flip(self, input_, spad, target, p=.5):
        """
        applies a random horizontal flip
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :param p: probability of applying the flip
        :return: flipped or original images
        """
        x = np.random.rand()
        if x < p:
            input_ = torch.flip(input_, (2,))
            spad = torch.flip(spad, (2,))
            target = torch.flip(target, (2,))

        return input_, spad, target

    def random_rotation(self, input_, spad, target, p=.25):
        """
        applies a random 90 degree rotation
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :param p: probability of applying the rotation
        :return: rotated or original images
        """
        x = np.random.rand()  # probability of rotation
        if x < p:
            input_ = torch.rot90(input_, 2, [1, 2])
            spad = torch.rot90(spad, 2, [1, 2])
            target = torch.rot90(target, 2, [1, 2])

        return input_, spad, target

    def check_files(self):
        """
        checks if the file extensions are correct
        :return: None
        """
        # check file extensions
        if not self.inputs[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected input images of type .hdr")
        if not self.spad_inputs[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected SPAD input images of type .hdr")
        if not self.targets[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected target images of type .hdr")

        return
