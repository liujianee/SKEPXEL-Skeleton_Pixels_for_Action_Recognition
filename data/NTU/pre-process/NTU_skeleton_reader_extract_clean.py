# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 12:00:57 2016

@author: Jian
"""

import re
import os
import numpy as np
import scipy.io as sio

WORK_PATH = '/media/jianl/disk3/Jian/Datasets/NTU'

dataPath = 'nturgb+d_skeletons'
savePath = 'nturgb+d_skeletons_extract'

cleanPath = 'nturgb+d_skeletons_extract_clean'
#swapPath = 'nturgb+d_skeletons_extract_swap'
swapPath = cleanPath

if not os.path.exists(cleanPath):
    os.makedirs(cleanPath)

if not os.path.exists(swapPath):
    os.makedirs(swapPath)

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


skel0_list = np.load('./skel0_list.npy') 
skel2_list = np.load('./skel2_list.npy')
skel3_list = np.load('./skel3_list.npy')

skel0_list = [x[1] for x in skel0_list]
skel2_list = [x[1] for x in skel2_list]
skel3_list = [x[1] for x in skel3_list]


ac_Files = os.listdir(savePath)
sort_nicely(ac_Files)

## step 1: to remove abnormal skeleton data

for ac_ix, one_ac in enumerate(ac_Files):
    print("index=%d, processing %s\n" % (ac_ix,one_ac))
    ac_id = one_ac.split('.')[0]
    src_file = os.path.join(WORK_PATH, savePath, one_ac)
    tgt_file = os.path.join(WORK_PATH, cleanPath, one_ac)
    os.symlink(src_file, tgt_file)

# 56880 ---> 56338
for skel0 in skel0_list:
    ac_id = skel0.split('.')[0]
    tgt_file = cleanPath + '/' + ac_id + '.mat'
    os.remove(tgt_file)

# 56338 ---> 56017
for skel3 in skel3_list:
    ac_id = skel3.split('.')[0]
    tgt_file = cleanPath + '/' + ac_id + '.mat'
    os.remove(tgt_file)

# 56017 ---> 54210
for skel2 in skel2_list:
    ac_id = skel2.split('.')[0]
    tgt_file = cleanPath + '/' + ac_id + '.mat'
    if os.path.exists(tgt_file):
        os.remove(tgt_file)

###########
## step 2: get rid of non-human object (i.e. chair) in those abnormal skeleton data, to make it normal
###########
def skeleton_swap(sk):
    sk_swap = np.zeros_like(sk)
    skel_ch_0 = sk[:,0:3,0]
    skel_ch_1 = sk[:,0:3,1]
    skel_ch_2 = sk[:,0:3,2]
    Y_range_ch_0 = max(skel_ch_0[:,1]) - min(skel_ch_0[:,1])
    Y_range_ch_1 = max(skel_ch_1[:,1]) - min(skel_ch_1[:,1])
    Y_range_ch_2 = max(skel_ch_2[:,1]) - min(skel_ch_2[:,1])
    Y_ranges = [Y_range_ch_0, Y_range_ch_1, Y_range_ch_2]
    sorted_idx = np.argsort(Y_ranges)[::-1]	# sort Y in descend order, return the index
    sk_swap[:,:,0] = sk[:,:,sorted_idx[0]]
    sk_swap[:,:,1] = sk[:,:,sorted_idx[1]]
    sk_swap[:,:,2] = sk[:,:,sorted_idx[2]]
    return sk_swap

# swap for skel3_list
for one_ac in skel3_list:
    if one_ac not in skel0_list:
	print("swapping for %s\n" % one_ac)
        ac_id = one_ac.split('.')[0]
        src_file = savePath + '/' + ac_id + '.mat'
        src_sk = sio.loadmat(src_file)['sk']
	tgt_sk = np.zeros_like(src_sk)
        fm_num = src_sk.shape[3]
        for fm_id in xrange(fm_num):
            tgt_sk[:,:,:,fm_id] = skeleton_swap(src_sk[:,:,:,fm_id])        
        tgt_file = swapPath + '/' + ac_id + '.mat' 
        sio.savemat(tgt_file, mdict={'sk':tgt_sk})


# swap for skel2_list
for one_ac in skel2_list:
    if one_ac not in skel0_list:
	print("swapping for %s\n" % one_ac)
        ac_id = one_ac.split('.')[0]
        src_file = savePath + '/' + ac_id + '.mat'
        src_sk = sio.loadmat(src_file)['sk']
	tgt_sk = np.zeros_like(src_sk)
        fm_num = src_sk.shape[3]
        for fm_id in xrange(fm_num):
            tgt_sk[:,:,:,fm_id] = skeleton_swap(src_sk[:,:,:,fm_id])        
        tgt_file = swapPath + '/' + ac_id + '.mat' 
        sio.savemat(tgt_file, mdict={'sk':tgt_sk})



