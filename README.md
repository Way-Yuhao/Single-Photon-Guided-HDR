# Single Photon Camera Guided HDR Imaging

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
tqdm
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

## 3. Usage

### Pretrained model
TODO: define pre-trained model

### Inference

```
python test.py --input <input/dir> --out <output/dir> --weights <weight/path>.pth 
```

