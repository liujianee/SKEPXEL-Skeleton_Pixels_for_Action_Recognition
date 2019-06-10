# to point PYTHONPATH to spams, run it before the rest codes
# "export PYTHONPATH=/usr/local/liblinear-2.11/python"

import os
import sys
import numpy as np
import scipy
import scipy.sparse as ssp
import h5py

from liblinear import *
from liblinearutil import *
import time

import scipy.io as sio
import operator

CLS_NUM = 60

param = '-c 4'

#work_dir_1 = '../SKEL_Output_SuperPixel_5x5_32_20170827-104321/'
#work_dir_1 = '../SKEL_Output_SuperPixel_5x5_32_20170826-213132/'
#work_dir_3 = './SKEL_Output_SuperPixel_5x5_32_SEED_1/'

#work_dir_1 = './SKEL_Output_COMPARE_Features_SuperPixel_5x5_36x36_Skeleton/'
work_dir_1 = './SKEL_Output_COMPARE_Features_SuperPixel_5x5_36x36_Skel_Velo_Sandwich/'
work_dir_2 = '/media/jianl/disk3/Jian/Datasets/NTU/CLASSIFICATION/HPM_Descriptors_GoogLeNet_GAN/HPM_Output/'
work_dir_3 = './SKEL_Output_COMPARE_Features_SuperPixel_5x5_36x36_Skeleton/'

skel_secDirs_1 = os.listdir(work_dir_1)
skel_secDirs_2 = os.listdir(work_dir_2)
skel_secDirs_3 = os.listdir(work_dir_3)

skel_secDirs_1.sort()
skel_secDirs_2.sort()
skel_secDirs_3.sort()

skel_batches_1 = skel_secDirs_1[0:5]     	## config here
skel_batches_2 = skel_secDirs_2[0:5]     	## config here
skel_batches_3 = skel_secDirs_3[0:5]     	## config here

score = []
#X_multibatch = np.zeros([1, 1792*28*2])
X_multibatch = np.zeros([1, (1792+1024)*28])
Y_multibatch = np.zeros([1, 5])

#tX_multibatch = np.zeros([1, 1792*28*2])
tX_multibatch = np.zeros([1, (1792+1024)*28])
tY_multibatch = np.zeros([1, 5])

for (skel_sec_1, skel_sec_2, skel_sec_3) in zip(skel_batches_1, skel_batches_2, skel_batches_3):
    print("processing %s + %s\n" % (skel_sec_1, skel_sec_2))

    skel_featureDir = work_dir_1 + skel_sec_1
    matfile1 = skel_featureDir + '/trainData_view.mat'
    mat = h5py.File(matfile1)
    X_skel_1 = np.asarray(mat['trainData']).transpose()

    matfile2 = skel_featureDir + '/trainLab_view.mat'
    mat = sio.loadmat(matfile2)
    lab_skel_1 = mat['trainLab']


    skel_featureDir = work_dir_2 + skel_sec_2
    matfile1 = skel_featureDir + '/trainData_view.mat'
    mat = h5py.File(matfile1)
    X_skel_2 = np.asarray(mat['trainData']).transpose()

    matfile2 = skel_featureDir + '/trainLab_view.mat'
    mat = sio.loadmat(matfile2)
    lab_skel_2 = mat['trainLab']


    skel_featureDir = work_dir_3 + skel_sec_3
    matfile1 = skel_featureDir + '/trainData_view.mat'
    mat = h5py.File(matfile1)
    X_skel_3 = np.asarray(mat['trainData']).transpose()

    matfile2 = skel_featureDir + '/trainLab_view.mat'
    mat = sio.loadmat(matfile2)
    lab_skel_3 = mat['trainLab']


    if (lab_skel_1 == lab_skel_2).all():
        print("Matched! combining...")
        X = np.concatenate((X_skel_1, X_skel_2), axis=1)
    if (lab_skel_1 != lab_skel_2).any():
        print("ERROR! trainData Matching fail!")
        break

    X_multibatch = np.concatenate((X_multibatch, X), axis=0)
    Y_multibatch = np.concatenate((Y_multibatch, lab_skel_1), axis=0)

X = X_multibatch[1:,:]
Y = Y_multibatch[1:,:][:,-1]

model = train(Y, X, param)

for (skel_sec_1, skel_sec_2, skel_sec_3) in zip(skel_batches_1, skel_batches_2, skel_batches_3):
    print("testing %s + %s\n" % (skel_sec_1, skel_sec_2))

    skel_featureDir = work_dir_1 + skel_sec_1
    matfile3 = skel_featureDir + '/testData_view.mat'
    mat = h5py.File(matfile3)
    testData_skel_1 = np.asarray(mat['testData']).transpose()

    matfile4 = skel_featureDir + '/testLab_view.mat'
    mat = sio.loadmat(matfile4)
    testLab_skel_1 = mat['testLab']


    skel_featureDir = work_dir_2 + skel_sec_2
    matfile3 = skel_featureDir + '/testData_view.mat'
    mat = h5py.File(matfile3)
    testData_skel_2 = np.asarray(mat['testData']).transpose()

    matfile4 = skel_featureDir + '/testLab_view.mat'
    mat = sio.loadmat(matfile4)
    testLab_skel_2 = mat['testLab']


    skel_featureDir = work_dir_3 + skel_sec_3
    matfile3 = skel_featureDir + '/testData_view.mat'
    mat = h5py.File(matfile3)
    testData_skel_3 = np.asarray(mat['testData']).transpose()

    matfile4 = skel_featureDir + '/testLab_view.mat'
    mat = sio.loadmat(matfile4)
    testLab_skel_3 = mat['testLab']

    if (testLab_skel_1 == testLab_skel_2).all():
        print("Matched! testing...")
        testData = np.concatenate((testData_skel_1, testData_skel_2), axis=1)
    if (testLab_skel_1 != testLab_skel_2).any():
        print("ERROR! testData Matching fail!")
        break

    tX_multibatch = np.concatenate((tX_multibatch, testData), axis=0)
    tY_multibatch = np.concatenate((tY_multibatch, testLab_skel_1), axis=0)

testData = tX_multibatch[1:,:]
testLab = tY_multibatch[1:,:][:,-1]

p_label, p_acc, p_val = predict(testLab, testData, model)

print("Score is %.2f\n" % p_acc[0])

score_arr = np.asarray(p_acc[0])

#np.save('liblinear_SVM_score_XView_allbatch', score_arr)

