# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 12:00:57 2016

@author: Jian
"""

import re
import os
import numpy as np
import scipy.io as sio

dataPath = 'nturgb+d_skeletons'
savePath = 'nturgb+d_skeletons_extract'

if not os.path.exists(savePath):
    os.makedirs(savePath)

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


# sort pose dir to add class ID accordingly
ac_Files = os.listdir(dataPath)
sort_nicely(ac_Files)

skel0_list = []
skel2_list = []
skel3_list = []

for ac_ix, one_ac in enumerate(ac_Files):
    print("index=%d, processing %s\n" % (ac_ix,one_ac))
    ac_id = one_ac.split('.')[0]
    skel_file = dataPath + '/' + one_ac
    f = open(skel_file, 'r')
    f_lines = f.readlines()
    f.close()
    skel_arr = np.zeros([25,6,3,1])	#25-joint number; 6-info columns; 3-maximum skeletons; 1-frame
    ff_list = []
    zero_skeletons = False	# mark the action where there is missing skeleton
    two_skeletons = False	# mark the action where fake skeleton is detected
    three_skeletons = False
    for ix, line in enumerate(f_lines):
        if (line == '0\r\n'):
    	    zero_skeletons = True
            ff_arr = np.zeros([25,6,3,1]);
            skel_arr = np.concatenate((skel_arr, ff_arr), axis=3)
        if (line == '1\r\n'):
            ff_arr = np.zeros([25,6,3,1]);    
            ff = f_lines[ix+3 : ix+28]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,0,0] = np.asarray(ff_list).astype(float)
            skel_arr = np.concatenate((skel_arr, ff_arr), axis=3)
        if (line == '2\r\n'):
            two_skeletons = True
            ff_arr = np.zeros([25,6,3,1]);    
            ff = f_lines[ix+3 : ix+28]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,0,0] = np.asarray(ff_list).astype(float)
            ff = f_lines[ix+30 : ix+55]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,1,0] = np.asarray(ff_list).astype(float)
            skel_arr = np.concatenate((skel_arr, ff_arr), axis=3)
        if (line == '3\r\n'):
	    three_skeletons = True
            ff_arr = np.zeros([25,6,3,1]);    
            ff = f_lines[ix+3 : ix+28]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,0,0] = np.asarray(ff_list).astype(float)
            ff = f_lines[ix+30 : ix+55]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,1,0] = np.asarray(ff_list).astype(float)
            ff = f_lines[ix+57 : ix+82]
            ff_list = [fff.split(' ')[0:6] for fff in ff]
            ff_arr[:,:,2,0] = np.asarray(ff_list).astype(float)
            skel_arr = np.concatenate((skel_arr, ff_arr), axis=3)
    if zero_skeletons:
        skel0_list.append((ac_ix, one_ac))
    if two_skeletons:
        skel2_list.append((ac_ix, one_ac))
    if three_skeletons:
        skel3_list.append((ac_ix, one_ac))

    matfile = savePath + '/' + ac_id + '.mat'
    sio.savemat(matfile, {'sk': skel_arr[:,:,:,1:]})

####
# filter skel2_list to get rid of action with label A050~A060
####


np.save('./skel0_list.npy', skel0_list)
np.save('./skel2_list.npy', skel2_list)
np.save('./skel3_list.npy', skel3_list)




