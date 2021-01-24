import cv2
import numpy as np

CMOS_qe = {                 # quantum efficiency index for each color channel
    'r' : .40,
    'g' : .75,
    'b' : .77
}

print(CMOS_qe['r'])