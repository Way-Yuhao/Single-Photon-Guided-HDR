# [WACV 2022] Single Photon Camera Guided HDR Imaging

[![Project Webpage](https://img.shields.io/badge/Project-Webpage-0A66C2)](https://yuhaoliu.net/projects/spc_guided_hdr.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Conference Paper | WACV (2022)](https://img.shields.io/badge/Conference%20Paper-WACV%20(2022)-4B5563)](https://openaccess.thecvf.com/content/WACV2022/html/Liu_Single-Photon_Camera_Guided_Extreme_Dynamic_Range_Imaging_WACV_2022_paper.html)

## 1. Getting Started

Clone the repo:

  ```bash
  git clone https://github.com/Way-Yuhao/Single-Photon-Guided-HDR.git
  ```

## 2. Requirements

Install dependencies from `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

### Input file directory structure

The input directory passed to `--input` should contain paired CMOS and SPAD image folders:

```text
input/
├── CMOS/
├── SPAD/
└── ideal/  # optional ground truth images
```

Both CMOS and SPAD inputs are required. A ground truth (ideal) image can be added to compute the loss of network output.

### CMOS & SPAD image requirements
* To properly use our provided ```test.py``` testing script, it is required to know the constant CMOS saturation limit for all CMOS images. This value is used to normalize input data for the neural network. 

## 3. Usage

### Pretrained model
Please [fill out this form](https://forms.gle/Xo2qL99vpChxLTtx8) to obtain link to our model weights (Google Drive).

### Inference

```
python test.py --input <input/dir> --out <output/dir> --weights <weight/path>.pth --saturation <saturation/float>
```

Parameters and their descriptions:
>```input```: directory of input images.<br/>
>```out```: path to output directory.<br/>
>```weights```: path to the pretrained neural network weights.<br/>
>```cpu```: toggle to use CPU only.<br/>
>```saturation```: an int or float defining CMOS saturation limit.<br/>
>```gain```: a float defining gain applied to SPAD images during data loading stage; for debugging only.<br/>
>```experimental```: toggle if testing on experimental data / no ground truth provided.<br/>
>```plot```: toggle if intend to use plt.show() to visualize outputs.<br/>

## 4. Citation
If you find this useful, please cite
```
@InProceedings{Liu_2022_WACV,
    author    = {Liu, Yuhao and Gutierrez-Barragan, Felipe and Ingle, Atul and Gupta, Mohit and Velten, Andreas},
    title     = {Single-Photon Camera Guided Extreme Dynamic Range Imaging},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1575-1585}
}
```
## 5. Acknowledgment 

This work was supported in part by the U.S. Department of Energy/NNSA (DE-NA0003921), National Science Foundation (CAREER 1846884 and 1943149), and UW-Madison’s Draper Technology Innovation Fund. The authors would like to thank the Computational Imaging Group at Rice University for providing conference travel funds for Yuhao Liu. U.S. DoE full legal disclaimer: https://www.osti.gov/stip/about/disclaimer.

We used a standard unet (https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) as a starting point for our network. Our latest version does not contain any code from the standard unet.
