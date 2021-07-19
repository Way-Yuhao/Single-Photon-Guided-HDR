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

### Input file directory structure
>$input directory$<br/>
>>CMOS<br/>
>>SPAD<br/>
>>ideal (optional)<br/>

Both CMOS and SPAD inputs are required. A ground truth (ideal) image can be added to compute the loss of network output.

### CMOS & SPAD image requirements
* To properly use our provided ```test.py``` testing script, it is required to know the constant CMOS saturation limit for all CMOS images. This value is used to normalize input data for the neural network. 

## 3. Usage

### Pretrained model
TODO: define pre-trained model

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

## Contact
TODO
## License
TODO
