import torch
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler


class SubsetSequenceSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
        # self.ptr = 0

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))
        # return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)