from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import os
import os.path
import numpy as np
import sys
from natsort import natsorted
from matplotlib import pyplot as plt
from pytorch_run import disp_plt, disp_sample, CMOS_sat

# IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
#
#
# def has_file_allowed_extension(filename, extensions):
#     """Checks if a file is an allowed extension.
#
#     Args:
#         filename (string): path to a file
#         extensions (tuple of strings): extensions to consider (lowercase)
#
#     Returns:
#         bool: True if the filename ends with one of given extensions
#     """
#     return filename.lower().endswith(extensions) | filename.lower().endswith("hdr")
#
#
# def is_image_file(filename):
#     """Checks if a file is an allowed image extension.
#
#     Args:
#         filename (string): path to a file
#
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     return has_file_allowed_extension(filename, IMG_EXTENSIONS)
#
#
# def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
#     images = []
#     dir = os.path.expanduser(dir)
#     if not ((extensions is None) ^ (is_valid_file is None)):
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
#     if extensions is not None:
#         def is_valid_file(x):
#             return has_file_allowed_extension(x, extensions)
#     for target in sorted(class_to_idx.keys()):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in natsorted(fnames, number_type=int):
#                 path = os.path.join(root, fname)
#                 if is_valid_file(path):
#                     item = (path, class_to_idx[target])
#                     images.append(item)
#
#     return images
#
#
# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')
#
#
# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)
#
# def default_loader(path):
#     return cv_loader(path)



def cv_loader(path):
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    return img


def down_sample(input_, target, down_sp_rate):
    """
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


def normalize(input_, spad, target):
    """
    normalizes input to [0, 1], spad and target to [0, >1]
    :param input_: input tensor. Expects original input image files to be 16-bit PNG (uint16)
    :param spad: side input tensor. Expects original input image files to be 32-bit .hdr (float32)
    :param target: target tensor. Expects original label image files to be 32-bit .hdr (float32)
    :return: normalized input and label
    """
    input_ = input_ / CMOS_sat
    spad = spad / CMOS_sat
    target = target / CMOS_sat
    return input_, spad, target


def random_crop(input_, spad, target):
    """

    :param input_:
    :param spad:
    :param target:
    :return:
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
        h = np.random.randint(0, max_h/2) * 2  # random even numbers
        w = np.random.randint(0, max_w/2) * 2
        input_crop = input_[:, h: h + crop_height, w: w + crop_width]
        target_crop = target[:, h: h + crop_height, w: w + crop_width]
        spad_crop = spad[:, int(h/diff): int((h + crop_height)/diff), int(w/diff): int((w + crop_width)/diff)]
        if torch.sum(input_crop > CMOS_sat) > sat_bound:
            is_saturated = True
        else:
            counter -= 1

    return input_crop, spad_crop, target_crop


def random_horizontal_flip(input_, spad, target, p=.5):
    x = np.random.rand()
    if x < p:
        input_ = torch.flip(input_, (2,))
        spad = torch.flip(spad, (2,))
        target = torch.flip(target, (2,))

    return input_, spad, target


def random_rotation(input_, spad, target, p=.25):
    x = np.random.rand()  # probability of rotation
    if x < p:
        input_ = torch.rot90(input_, 2, [1, 2])
        spad = torch.rot90(spad, 2, [1, 2])
        target = torch.rot90(target, 2, [1, 2])

    return input_, spad, target


def data_augmentation(input_, spad, target):

    input_, spad, target = random_crop(input_, spad, target)
    input_, spad, target = random_horizontal_flip(input_, spad, target)
    input_, spad, target = random_rotation(input_, spad, target)

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

    def __init__(self, input_dir, spad_dir, target_dir, input_transform=None, spad_transform=None, target_transform=None):

        # removed number_type=int
        self.inputs = natsorted(os.listdir(input_dir))
        self.targets = natsorted(os.listdir(target_dir))
        self.spad_inputs = natsorted(os.listdir(spad_dir))
        self.input_dir = input_dir
        self.spad_dir = spad_dir
        self.target_dir = target_dir
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
        return

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input_sample = cv_loader(self.input_dir + self.inputs[item])
        spad_sample = cv_loader(self.spad_dir + self.spad_inputs[item])
        target_sample = cv_loader(self.target_dir + self.targets[item])
        input_sample = self.input_transform(input_sample)
        spad_sample = self.spad_transform(spad_sample)
        target_sample = self.target_transform(target_sample)
        input_sample, spad_sample, target_sample = data_augmentation(input_sample, spad_sample, target_sample)
        input_sample, spad_sample, target_sample = normalize(input_sample, spad_sample, target_sample)

        # spad_sample = spad_sample[1, :, :].unsqueeze(dim=0)  # only keep one channel

        # TODO: remove this (monochrome)
        # input_sample = input_sample[1, :, :]
        # input_sample = torch.stack((input_sample, input_sample, input_sample), dim=0)
        # target_sample = target_sample[1, :, :]
        # target_sample = torch.stack((target_sample, target_sample, target_sample), dim=0)

        return input_sample, spad_sample, target_sample

    def check_files(self):
        # check file extensions
        if not self.inputs[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected input images of type .png")
        if not self.spad_inputs[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected SPAD input images of type .hdr")
        if not self.targets[0].lower().endswith("hdr"):
            raise FileExistsError("ERROR: expected target images of type .hdr")

        return


# class DatasetFolder(VisionDataset):
#     """A generic data loader where the samples are arranged in this way: ::
#
#         root/class_x/xxx.ext
#         root/class_x/xxy.ext
#         root/class_x/xxz.ext
#
#         root/class_y/123.ext
#         root/class_y/nsdf3.ext
#         root/class_y/asd932_.ext
#
#     Args:
#         root (string): Root directory path.
#         loader (callable): A function to load a sample given its path.
#         extensions (tuple[string]): A list of allowed extensions.
#             both extensions and is_valid_file should not be passed.
#         transform (callable, optional): A function/transform that takes in
#             a sample and returns a transformed version.
#             E.g, ``transforms.RandomCrop`` for images.
#         target_transform (callable, optional): A function/transform that takes
#             in the target and transforms it.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid_file (used to check of corrupt files)
#             both extensions and is_valid_file should not be passed.
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         samples (list): List of (sample path, class_index) tuples
#         targets (list): The class_index value for each image in the dataset
#     """
#
#     def __init__(self, root, loader, extensions=None, transform=None,
#                  target_transform=None, is_valid_file=None):
#         super(DatasetFolder, self).__init__(root, transform=transform,
#                                             target_transform=target_transform)
#         classes, class_to_idx = self._find_classes(self.root)
#         samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
#         if len(samples) == 0:
#             raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
#                                 "Supported extensions are: " + ",".join(extensions)))
#
#         self.loader = loader
#         self.extensions = extensions
#
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]
#
#     def _find_classes(self, dir):
#         """
#         Finds the class folders in a dataset.
#
#         Args:
#             dir (string): Root directory path.
#
#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#
#         Ensures:
#             No class is a subdirectory of another.
#         """
#         if sys.version_info >= (3, 5):
#             # Faster and available in Python 3.5 and above
#             classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#         else:
#             classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#         classes.sort()
#         class_to_idx = {classes[i]: i for i in range(len(classes))}
#         return classes, class_to_idx
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.input_transform is not None:
#             sample = self.input_transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return sample, target
#
#     def __len__(self):
#         return len(self.samples)

