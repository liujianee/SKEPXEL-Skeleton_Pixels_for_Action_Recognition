# to point PYTHONPATH to spams, run it before the rest codes
# "export PYTHONPATH=/usr/local/liblinear-2.30/python/"

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

CLS_NUM = 60

param = '-c 4'

work_dir = './SKEL_Output_SuperPixel_5x5_36_Skeleton/'

#FEATURE_W = 1024
FEATURE_W = 1792

secDirs = os.listdir(work_dir)
secDirs = secDirs

score = []
X_multibatch = np.zeros([1,FEATURE_W*28])
Y_multibatch = np.zeros([1, 5])

tX_multibatch = np.zeros([1,FEATURE_W*28])
tY_multibatch = np.zeros([1, 5])

secDirs = ['nturgb+d_skel_s001/', 'nturgb+d_skel_s002/', 'nturgb+d_skel_s003/']

for sec in secDirs:
    print("processing %s\n" % sec)
    featureDir = work_dir + sec
    matfile1 = featureDir + '/trainData_view.mat'
    mat = h5py.File(matfile1)
    X = np.asarray(mat['trainData']).transpose()
    #X = X/1
    matfile2 = featureDir + '/trainLab_view.mat'
    mat = sio.loadmat(matfile2)
    lab = mat['trainLab']
    #Y = np.zeros((len(lab), CLS_NUM))
    #for i in range(0,len(lab)):
    #    Y[i, (lab[i][4]-1)] = 1
    #Y = np.asfortranarray(Y)
    #X = np.asfortranarray(X)
    X_multibatch = np.concatenate((X_multibatch, X), axis=0)
    Y_multibatch = np.concatenate((Y_multibatch, lab), axis=0)


X = X_multibatch[1:,:]
Y = Y_multibatch[1:,:][:,-1]

#Y = np.asfortranarray(Y)
#X = np.asfortranarray(X)

model = train(Y, X, param)

#np.save('weight_matrix_XView_allbatch', W)
save_model('XView_model', model)

secDirs = ['nturgb+d_skel_s015/']
for sec in secDirs:
    print("testing %s\n" % sec)
    featureDir = work_dir + sec
    matfile3 = featureDir + '/testData_view.mat'
    mat = h5py.File(matfile3)
    testData = np.asarray(mat['testData']).transpose()
    matfile4 = featureDir + '/testLab_view.mat'
    mat = sio.loadmat(matfile4)
    testLab = mat['testLab']
    tX_multibatch = np.concatenate((tX_multibatch, testData), axis=0)
    tY_multibatch = np.concatenate((tY_multibatch, testLab), axis=0)

testData = tX_multibatch[1:,:]
testLab = tY_multibatch[1:,:][:,-1]

p_label, p_acc, p_val = predict(testLab, testData, model)

print("Score is %.2f\n" % p_acc[0])

score_arr = np.asarray(p_acc[0])

#np.save('liblinear_SVM_score_XView_Norm', score_arr)



