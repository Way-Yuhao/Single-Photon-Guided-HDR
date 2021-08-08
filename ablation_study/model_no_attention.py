from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from lum_fusion_model import ConvLayer, DeConvBlock, AttentionBlock, OneByOneConvBlock, SPADConvLayer
from ablation_study.model_no_spad import HDRNetNoSpad


class HDRNetNoAttention(nn.Module):
    def __init__(self, isMonochrome=False):
        super(HDRNetNoAttention, self).__init__()
        """flags"""
        self.isMonochrome = isMonochrome

        """Up Sampling and Luminance Fusion Network"""
        # layer depth #      0    1    2    3    4    5     6
        main_chs = np.array([3,  64, 128, 256, 512, 512, 1024])   # number of output channels for main encoder
        side_chs = np.array([-1,  3,  64,  128, 256, 512, -1])  # number of output channels for side encoder
        h =       np.array([128, 64,  32,  16,   8,   4,    2])   # height of tensors

        # encoder (VGG16 + extra Conv layer)
        self.vgg16 = models.vgg16(pretrained=True)
        encoded_features = list(self.vgg16.features)
        self.encoded_features = nn.ModuleList(encoded_features) # .eval()
        self.Conv6 = ConvLayer(in_ch=main_chs[5], out_ch=main_chs[6])

        # decoder
        self.DeConv6 = DeConvBlock(in_ch=main_chs[6], out_ch=main_chs[5])
        self.DeConv5 = DeConvBlock(in_ch=2 * main_chs[5] + side_chs[5], out_ch=main_chs[4])
        self.DeConv4 = DeConvBlock(in_ch=2 * main_chs[4] + side_chs[4], out_ch=main_chs[3])
        self.DeConv3 = DeConvBlock(in_ch=2 * main_chs[3] + side_chs[3], out_ch=main_chs[2])
        self.DeConv2 = DeConvBlock(in_ch=2 * main_chs[2] + side_chs[2], out_ch=main_chs[1])
        self.DeConv1 = DeConvBlock(in_ch=2 * main_chs[1], out_ch=main_chs[0])

        # attention gates
        # self.Att1 = AttentionBlock(F_g=main_chs[1], F_l=main_chs[1], F_int=main_chs[0])
        # self.Att0 = AttentionBlock(F_g=main_chs[0], F_l=main_chs[0], F_int=1)

        # spad encoder, with ReLU
        self.SpadConv2 = SPADConvLayer(in_ch=side_chs[1], out_ch=side_chs[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.SpadConv3 = SPADConvLayer(in_ch=side_chs[2], out_ch=side_chs[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.SpadConv4 = SPADConvLayer(in_ch=side_chs[3], out_ch=side_chs[4], kernel_size=2, stride=2, padding=0, bias=True)
        self.SpadConv5 = SPADConvLayer(in_ch=side_chs[4], out_ch=side_chs[5], kernel_size=2, stride=2, padding=0, bias=True)

        # final encoder: output # of channel is 3 for RGB, 1 for monochrome
        self.ConvOut = OneByOneConvBlock(in_ch=2 * main_chs[0], out_ch=main_chs[0] - isMonochrome * 2)

    def forward(self, x, y):
        # encoder
        encodings = []
        e = x
        for ii, model in enumerate(self.encoded_features):
            e = model(e)
            if ii in {4, 9, 16, 23, 30}:
                encodings.append(e)
        e1, e2, e3, e4, e5 = encodings
        e6 = self.Conv6(e5)

        # spad encoder
        y2 = self.SpadConv2(y)
        y3 = self.SpadConv3(y2)
        y4 = self.SpadConv4(y3)
        y5 = self.SpadConv5(y4)

        # decoder
        d5 = self.DeConv6(e6)
        d4 = self.DeConv5(d5, y5, e5)
        d3 = self.DeConv4(d4, y4, e4)
        d2 = self.DeConv3(d3, y3, e3)
        d1 = self.DeConv2(d2, y2, e2)
        # e1_att, _ = self.Att1(g=d1, x=e1)
        d0 = self.DeConv1(d1, e1)

        # final encodings
        # x_att, mask = self.Att0(g=d0, x=x)
        out = self.ConvOut(d0, x)
        return out
