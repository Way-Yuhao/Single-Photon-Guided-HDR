from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out, psi


class ConvLayer(nn.Module):
    """
    Convolution layer
    """

    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.conv(x)
        return x


class OneByOneConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OneByOneConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, d, e):
        x = torch.cat((e, d), dim=1)
        x = self.conv(x)
        return x


class DeConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DeConvBlock, self).__init__()
        self.resize = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)  # at discretion
        )

    def forward(self, d, y=None, e=None):
        """
        :param d: activation from previous de-conv layer
        :param y: encoded/unencoded SPAD tensor
        :param e: activation from long-skip connections
        :return:
        """
        if e is not None:
            d = torch.cat((e, d), dim=1)
        if y is not None:
            d = torch.cat((d, y), dim=1)
        # out = self.de(d)
        out = self.resize(d)
        return out


class SPADConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SPADConvLayer, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encode(x)


class IntensityGuidedHDRNet(nn.Module):
    def __init__(self, isMonochrome=False, outputMask=False):
        super(IntensityGuidedHDRNet, self).__init__()
        """flags"""
        self.isMonochrome = isMonochrome
        self.outputMask = outputMask

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
        self.Att1 = AttentionBlock(F_g=main_chs[1], F_l=main_chs[1], F_int=main_chs[0])
        self.Att0 = AttentionBlock(F_g=main_chs[0], F_l=main_chs[0], F_int=1)

        # OLD spad encoder, without ReLU
        # self.SpadConv2 = nn.Conv2d(side_chs[1], side_chs[2], kernel_size=1, stride=1, padding=0, bias=True)
        # self.SpadConv3 = nn.Conv2d(side_chs[2], side_chs[3], kernel_size=2, stride=2, padding=0, bias=True)
        # self.SpadConv4 = nn.Conv2d(side_chs[3], side_chs[4], kernel_size=2, stride=2, padding=0, bias=True)
        # self.SpadConv5 = nn.Conv2d(side_chs[4], side_chs[5], kernel_size=2, stride=2, padding=0, bias=True)

        # spad encoder, with ReLU
        self.SpadConv2 = SPADConvLayer(in_ch=side_chs[1], out_ch=side_chs[2])
        self.SpadConv3 = SPADConvLayer(in_ch=side_chs[2], out_ch=side_chs[3])
        self.SpadConv4 = SPADConvLayer(in_ch=side_chs[3], out_ch=side_chs[4])
        self.SpadConv5 = SPADConvLayer(in_ch=side_chs[4], out_ch=side_chs[5])

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
        e1_att, _ = self.Att1(g=d1, x=e1)
        d0 = self.DeConv1(d1, e1_att)

        # final encodings
        x_att, mask = self.Att0(g=d0, x=x)
        out = self.ConvOut(d0, x_att)

        # if mode outputMask is set to true, then visualize mask through output
        if self.outputMask is True:
            mask = torch.cat((mask, mask, mask), dim=1)
            out = mask

        return out
