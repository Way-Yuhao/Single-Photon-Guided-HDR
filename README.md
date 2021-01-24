# SPAD-Guided-HDR
Implementation of simulators for Single-photon avalanche diodes and CMOS sensors.

## Requirements
```
numpy~=1.18.5
matplotlib~=3.2.1
opencv-python~=4.2.0.34
scikit-image~=0.17.2
torch~=1.7.1
tqdm~=4.50.2
```
Install all dependent libraries:
  ```
  pip install -r requirements.txt
  ```


## Run the file
We require the file format for the HDR ground truth files to be .HDR. Note that .exr will not work with this implementation

Add all your folders to run_simulations.py:
```angular2
fpath =    # file path for input data 
out_path = # filepath for simulated output 
plt_path = #  filepath for histograms of ground truth, after some scaling
```
Then, twek the parameters for the SPAD and CMOS simulator, and run the scirpt via
```bash
python run _simulations.py
```