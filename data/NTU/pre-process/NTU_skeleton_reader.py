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
savePath = 'nturgb+d_skeletons_matfile'

if not os.path.exists(savePath):
    os.makedirs(savePath)

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


# sort pose dir to add class ID accordingly
ac_Files = os.listdir(dataPath)
sort_nicely(ac_Files)

for ix, one_ac in enumerate(ac_Files):
    print("index=%d, processing %s\n" % (ix,one_ac))
    ac_id = one_ac.split('.')[0]
    skel_file = dataPath + '/' + one_ac
    f = open(skel_file, 'r')
    f_lines = f.readlines()
    f.close()
    skel_arr = np.zeros([25,3])
    ff_list = []
    data_valid = False
    for ix, line in enumerate(f_lines):
        if (line == '1\r\n') or (line == '2\r\n') or (line == '3\r\n') or (line == '4\r\n'):
            ff = f_lines[ix+3 : ix+28]
            ff_list = [fff.split(' ')[0:3] for fff in ff]
            ff_arr = np.asarray(ff_list).astype(float)
            skel_arr = np.dstack((skel_arr, ff_arr))
            data_valid = True
        #
        # TODO: when there are more than one skeletons
        #
        #if (line == '0\r\n'): #TODO
        #    data_valid = False
    if data_valid:
        if (skel_arr.shape[2] > 24):	# to filter out the clips with less than 25 frames
            matfile = savePath + '/' + ac_id + '.mat'
            sio.savemat(matfile, {'sk': skel_arr[:,:,1:]})


