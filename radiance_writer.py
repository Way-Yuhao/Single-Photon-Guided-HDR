# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:17:21 2018

@author: ingle, yuhao
"""
import numpy as np

def radiance_writer(image, fname):
    """image should be a 3D matrix of RGB values in floating point numbers for each pixel"""
    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    mantissa, exponent = np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)
    rgbe[rgbe>255] = 255
    rgbe[rgbe<0] = 0
    rgbe = np.array(rgbe, dtype=np.uint8)

    f = open(fname, "wb")
    f.write("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n".encode())
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]).encode())
    rgbe.flatten().tofile(f)
    f.close()
    # print(fname, 'written.')


def radiance_writer_grayscale(image, fname):
    """image should be a 2D matrix of grayscale values in floating point numbers for each pixel"""
    brightest = image
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    mantissa, exponent = np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)  #TODO: 4?
    rgbe[..., 0:2] = np.around(image[..., 0:2] * scaled_mantissa[..., None])
    rgbe[..., 2] = np.around(exponent + 128)
    rgbe[rgbe > 255] = 255
    rgbe[rgbe < 0] = 0
    rgbe = np.array(rgbe, dtype=np.uint8)

    f = open(fname, "wb")
    f.write("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n".encode())
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]).encode())
    rgbe.flatten().tofile(f)
    f.close()
    # print(fname, 'written.')