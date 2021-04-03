# Unet-Segmentation-Pytorch-Nest-of-Unets

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-by-neckbeards.svg)](https://github.com/bigmb)

[![HitCount](http://hits.dwyl.io/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets.svg)](http://hits.dwyl.io/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/issues)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-a-nested-u-net-architecture-for-medical/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=unet-a-nested-u-net-architecture-for-medical)

Implementation of different kinds of Unet Models for Image Segmentation

1) **UNet** - U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

2) **RCNN-UNet** - Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
https://arxiv.org/abs/1802.06955

3) **Attention Unet** - Attention U-Net: Learning Where to Look for the Pancreas
https://arxiv.org/abs/1804.03999

4) **RCNN-Attention Unet** - Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)
<!--LeeJun Implementation - https://github.com/LeeJunHyun/Image_Segmentation.git -->

5) **Nested UNet** - UNet++: A Nested U-Net Architecture for Medical Image Segmentation
https://arxiv.org/abs/1807.10165

With Layer Visualization

## 1. Getting Started

Clone the repo:

  ```bash
  git clone https://github.com/Way-Yuhao/Single-Photon-Guided-HDR.git
  ```

## 2. Requirements

```
python>=3.6
torch>=0.4.0
torchvision
torchsummary
tensorboardx
natsort
numpy
pillow
scipy
scikit-image
sklearn
```
Install all dependent libraries:
  ```bash
  pip install -r requirements.txt
  ```


## Acknowledgement
The deep learning part of this work is partially based on https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets.git


