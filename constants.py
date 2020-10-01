# Computations
NUM_JOBS = 100
NUM_CORES = 1

# virtual SPAD properties
EXPOSURE_TIME = 5e-7  # 5e-3 for spad
DEAD_TIME = 150e-9
DARK_COUNT_RATE = 100

QE_SPAD_R = .15
QE_SPAD_G = .25
QE_SPAD_B = .4

# QE_SPAD_R = 1
# QE_SPAD_G = 1
# QE_SPAD_B = 1

# CCD properties
FWC = 5e-3 / DEAD_TIME
QE_CCD_R = .40
QE_CCD_G = .75
QE_CCD_B = .77

# Image, before crop or down sampling
WIDTH = 1024
HEIGHT = 512
