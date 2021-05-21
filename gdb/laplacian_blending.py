#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import autoreload

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import cv2
import skimage.transform as skt # I have skimage v0.17.2
import skimage.morphology as skm
from scipy.sparse.linalg import spsolve
import skimage.filters as skf


# In[3]:


grtr_img = cv2.imread('hdr_images/435_gt.hdr', -1)
spad_img = cv2.imread('hdr_images/435_spad.hdr', -1)
cmos_img = cv2.imread('hdr_images/435_cmos.png', -1)
cmos_img = np.array(cmos_img, dtype=np.float32)
print(cmos_img.shape, spad_img.shape)

print(np.min(cmos_img.flatten()), np.min(spad_img.flatten()))
print(np.max(cmos_img.flatten()), np.max(spad_img.flatten()))


# In[4]:


# create our binary mask. 1 mean cmos image is saturated so include spad info at those pixels.
mask3 = np.zeros_like(cmos_img)
mask3[cmos_img>=65535] = 1
mask3 = np.array(mask3, dtype=np.bool)


# In[5]:


# naive upscale 4x the SPAD image so it's the same dimensions as CMOS image
#TODO later: replace this with a DNN upsampler like ExpandNet or something for a smarter super-resolution upsampling

spad_img_py_up = skt.pyramid_expand(spad_img,cmos_img.shape[0]//spad_img.shape[0],multichannel=True, preserve_range=True)

#spad_img_upsampled = broadcast_tile(spad_img[:,:,0], 4,4)
#plt.subplot(2,1,1)
#plt.imshow(np.log10(spad_img[:,:,0]+1e-12), cmap='gray')
#plt.subplot(2,1,2)
#plt.imshow(np.log10(spad_img_upsampled+1e-12), cmap='gray')


# In[17]:


num_layers = 4

cmos_pyr_lapl = list(skt.pyramid_laplacian(cmos_img, max_layer=num_layers,multichannel=True, preserve_range=True))
cmos_pyr_gaus = list(skt.pyramid_gaussian(cmos_img, max_layer=num_layers,multichannel=True, preserve_range=True))

spad_pyr_lapl = list(skt.pyramid_laplacian(spad_img_py_up, max_layer=num_layers,multichannel=True, preserve_range=True))
spad_pyr_gaus = list(skt.pyramid_gaussian(spad_img_py_up, max_layer=num_layers,multichannel=True, preserve_range=True))


mask_pyr_gaus = list(skt.pyramid_gaussian(mask3, max_layer=num_layers,multichannel=True, preserve_range=True))


# In[18]:


for cpl in cmos_pyr_lapl:
    print(cpl.shape)
print('----')
for cpg in cmos_pyr_gaus:
    print(cpg.shape)
print('----')
for mpg in mask_pyr_gaus:
    print(mpg.shape)


# In[19]:


# blended laplacian pyramid has edge information from both images blended with the mask values at each pyramid level
combined_lapl = list()
for i in range(len(cmos_pyr_lapl)):
    cur_comb_lapl = np.multiply(cmos_pyr_lapl[i], 1-mask_pyr_gaus[i]) + np.multiply(spad_pyr_lapl[i],mask_pyr_gaus[i])
    combined_lapl.append(cur_comb_lapl)


# In[20]:


# reconstruct the blended image working our way up to larger planes in the pyramid

prev_level = np.multiply(cmos_pyr_gaus[num_layers], 1-mask_pyr_gaus[num_layers]) + \
             np.multiply(spad_pyr_gaus[num_layers], mask_pyr_gaus[num_layers]) + combined_lapl[num_layers]

for lyr in range(num_layers-1, -1, -1): # e.g. if num_layers=6 count down 5,4,3,2,1,0
    cur_level = skt.pyramid_expand(prev_level,multichannel=True, preserve_range=True) +\
                combined_lapl[lyr]
    prev_level = np.array(cur_level)


# In[23]:


from radiance_writer import *
radiance_writer(cur_level[:,:,::-1], 'blended.hdr')
# radiance_writer writes BGR format so we feed in our image in color-reversed order
# so that the final output is effectively RGB for photoshop convenience


# In[21]:


plt.subplot(311)
plt.imshow(np.log10(cur_level[:,:,0]),cmap='gray')
plt.axis('off')
plt.subplot(312)
plt.imshow(np.log10(cur_level[:,:,1]),cmap='gray')
plt.axis('off')
plt.subplot(313)
plt.imshow(np.log10(cur_level[:,:,2]),cmap='gray')
plt.axis('off')
plt.tight_layout()

