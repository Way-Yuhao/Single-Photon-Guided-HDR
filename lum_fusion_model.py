from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Convolution Block without BatchNorm
    """

    def __init__(self, in_ch, out_ch, f,):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


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
        return out


class LumFusionNet(nn.Module):
    """
    Attention U-Net implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(LumFusionNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = ConvBlock(filters[4] + 4, filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = ConvBlock(filters[3] + 1, filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.SpadConv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.SpadConv2 = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x, y):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        y1 = self.SpadConv1(y)
        y2 = self.SpadConv2(y1)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5, y2), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4, y1), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = F.relu(out)

        return out


class ConvLayer(nn.Module):
    """
    Convolution Block
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

    def __init__(self, in_ch, out_ch, output_size, f=3):
        super(DeConvBlock, self).__init__()

        # de_conv_temp = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=f, stride=2, padding=1)
        # de_conv_layer = DeConvLayer(de_conv_temp, output_size=output_size)
        de_conv_layer = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.de = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            de_conv_layer,
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True))  # at discretion

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
        out = self.de(d)
        return out


class DeConvLayer(nn.Module):

    def __init__(self, conv, output_size):
        super(DeConvLayer, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


def _split_chs(img):
    """

    :param img: tensor of shape (m, 3, h, w)
    :return: b, g, r
    """
    b = img[:, 0, :, :].unsqueeze(1)
    g = img[:, 1, :, :].unsqueeze(1)
    r = img[:, 2, :, :].unsqueeze(1)
    return b, g, r


def _stack_chs(b, g, r):
    b = b.squeeze()
    g = g.squeeze()
    r = r.squeeze()
    out = torch.stack((b, g, r), dim=1)
    return out


class IntensityGuidedHDRNet(nn.Module):
    def __init__(self):
        super(IntensityGuidedHDRNet, self).__init__()

        """Up Sampling and Luminance Fusion Network"""
        # layer depth #      0    1    2    3    4    5     6
        main_chs = np.array([3,  64, 128, 256, 512, 512, 1024])   # number of output channels for main encoder
        side_chs = np.array([-1,  3,   4,  16,  64, 128,   -1])   # number of output channels for side encoder
        h =       np.array([128, 64,  32,  16,   8,   4,    2])   # height of tensors

        # encoder (VGG16 + extra Conv layer)
        self.vgg16 = models.vgg16(pretrained=True)
        encoded_features = list(self.vgg16.features)
        self.encoded_features = nn.ModuleList(encoded_features).eval()
        self.Conv6 = ConvLayer(in_ch=main_chs[5], out_ch=main_chs[6])

        # decoder
        self.DeConv6 = DeConvBlock(in_ch=main_chs[6], out_ch=main_chs[5], output_size=(h[5], h[5] * 2))
        self.DeConv5 = DeConvBlock(in_ch=2 * main_chs[5] + side_chs[5], out_ch=main_chs[4], output_size=(h[4], h[4] * 2))
        self.DeConv4 = DeConvBlock(in_ch=2 * main_chs[4] + side_chs[4], out_ch=main_chs[3], output_size=(h[3], h[3] * 2))
        self.DeConv3 = DeConvBlock(in_ch=2 * main_chs[3] + side_chs[3], out_ch=main_chs[2], output_size=(h[2], h[2] * 2))
        self.DeConv2 = DeConvBlock(in_ch=2 * main_chs[2] + side_chs[2], out_ch=main_chs[1], output_size=(h[1], h[1] * 2))
        self.DeConv1 = DeConvBlock(in_ch=2 * main_chs[1], out_ch=main_chs[0], output_size=(h[0], h[0] * 2))

        # attention gates
        self.Att1 = AttentionBlock(F_g=main_chs[1], F_l=main_chs[1], F_int=main_chs[0])
        self.Att0 = AttentionBlock(F_g=main_chs[0], F_l=main_chs[0], F_int=1)

        # spad encoder
        self.SpadConv2 = nn.Conv2d(side_chs[1], side_chs[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.SpadConv3 = nn.Conv2d(side_chs[2], side_chs[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.SpadConv4 = nn.Conv2d(side_chs[3], side_chs[4], kernel_size=2, stride=2, padding=0, bias=True)
        self.SpadConv5 = nn.Conv2d(side_chs[4], side_chs[5], kernel_size=2, stride=2, padding=0, bias=True)

        # final encoders
        self.ConvOut = OneByOneConvBlock(in_ch=2 * main_chs[0], out_ch=main_chs[0])

    def forward(self, x, y):
        # split color channels
        # x_b, x_g, x_r = _split_chs(x)
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
        e1_att = self.Att1(g=d1, x=e1)
        d0 = self.DeConv1(d1, e1_att)

        # final encodings
        x_att = self.Att0(g=d0, x=x)
        out = self.ConvOut(d0, x_att)
        return out
