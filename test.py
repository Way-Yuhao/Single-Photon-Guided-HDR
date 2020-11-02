import numpy as np
import cv2
import os
path = "./input/53 HDRI/"
for filename in os.listdir(path):
    print(os.path.join(path, filename))
    img = cv2.imread(os.path.join(path, filename), -1)
    cv2.imshow(img)
    break