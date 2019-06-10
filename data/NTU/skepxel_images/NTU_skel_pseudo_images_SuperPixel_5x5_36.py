
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from tqdm import tqdm
import random
import cv2


SPIXEL = 5
SPATIAL_DIM = 36
TEMPORAL_DIM = 36
STRIDE = 1	# decide how many pseudo images to be created
SKIP = 1	# decide how dense/sparse the skeleton frames are sampled, to build one pseudo image


dataPath = '/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/'

savePath = './nturgb+d_skeletons_pseudo_image_SuperPixel_5x5_36/'


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

if not os.path.exists(savePath):
    os.makedirs(savePath)

datafiles = glob.glob(dataPath + '*.mat')
sort_nicely(datafiles)


def skel_interpolate(skel_norm):
  fm_num = skel_norm.shape[2]
  skel_dim = skel_norm.shape[0]
  intep_skel = np.zeros([skel_dim, 3, fm_num*2 - 1])
  for ix in range(fm_num*2 - 1):
    if ix % 2 ==0:
      intep_skel[:,:,ix] = skel_norm[:,:,int(ix/2)]
    else:
      intep_skel[:,:,ix] = (skel_norm[:,:,int(ix/2)] + skel_norm[:,:,int(ix/2)+1])/2
  return intep_skel


def super_pixel(skel_frame, random_seed):
  random.seed(random_seed)
  joints_order = np.reshape(random.sample(xrange(25), 25),(5,5))
  skel_spixel = skel_frame[joints_order]
  return skel_spixel

for skel_file in tqdm(datafiles[46606:49346]):
  skel_norm = sio.loadmat(skel_file)
  skel_norm = skel_norm['sk']
  ac_id = skel_file.split('/')[-1].split('.')[0]
  s_batch = 'nturgb+d_skel_s' + ac_id.split('C0')[0].split('S')[1]
  save_dir = os.path.join(savePath, s_batch, ac_id)
  fm_num = skel_norm.shape[2]
  if fm_num < TEMPORAL_DIM:
      skel_norm = skel_interpolate(skel_norm)
      fm_num = skel_norm.shape[2]
      if fm_num < TEMPORAL_DIM:
          skel_norm = skel_interpolate(skel_norm)
          fm_num = skel_norm.shape[2]
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  img_num = (fm_num - TEMPORAL_DIM*SKIP)/STRIDE + 1
  for img_ix in xrange(img_num):
    skel_arr = np.zeros((SPATIAL_DIM*SPIXEL,TEMPORAL_DIM*SPIXEL,3), dtype=float)
    for frame_ix in xrange(TEMPORAL_DIM):
      current_frame = skel_norm[:,:,(img_ix*STRIDE + frame_ix*SKIP)]
      for order_ix in xrange(SPATIAL_DIM):
        skel_arr[order_ix*SPIXEL : (order_ix+1)*SPIXEL, frame_ix*SPIXEL : (frame_ix+1)*SPIXEL] = super_pixel(current_frame, order_ix)

    save_file = save_dir + '/' + '{:08}'.format(img_ix+1) + '.png'

    skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
    skel_img = np.array(skel_img * 255, dtype = np.uint8)
    #cv2.imwrite(save_file, skel_img)

    ## ATTENTION: cv2.imwrite will swap channels to BGR and then save!!
    ## if saved with cv2.imwrite, and then read with scipy.misc.read, the array not equal (RGB--BGR)
    ## it's better to write and read with the same library!

    #matplotlib.image.imsave(save_file, skel_arr)	# No_normalization
    scipy.misc.imsave(save_file, skel_img)



