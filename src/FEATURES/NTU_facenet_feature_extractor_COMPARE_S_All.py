# to run this code, must export PYTHONPATH like below:
# export PYTHONPATH=/media/jianl/TOSHIBA-EXT/Projects/facenet/src

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import re
import glob
import scipy.io as sio
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse

TEMPORAL_DIM = 36
SPATIAL_DIM = 36

SPIXEL = 5
STRIDE = 1	# decide how many pseudo images to be created
SKIP = 1	# decide how dense/sparse the skeleton frames are sampled, to build one pseudo image

save_dir = './SKEL_Features/'


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


def get_image_paths_and_labels(dataPath=None, s_ID=None):
    image_list = []
    label_list = []
    datafiles = glob.glob(dataPath + '*.mat')
    sort_nicely(datafiles)
    if s_ID == 15:
        ac_batch = datafiles[46606:49346]
    else:
        ac_batch = datafiles	# TODO
    for one_ac in ac_batch:
        view_ID = int(one_ac.split('/')[-1].split('P')[0].split('C')[1])
        sub_ID = int(one_ac.split('/')[-1].split('R')[0].split('P')[1])
        action_ID = int(one_ac.split('/')[-1].split('.')[0].split('A')[1]) - 1
        ac_matfile = one_ac
        image_list.append(ac_matfile)
        label_list.append(action_ID)
    return image_list, label_list


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


def create_image_out_of_skeleton(skel_norm, TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx):
    fm_num = skel_norm.shape[2]
    img_num = int((fm_num - TEMPORAL_DIM*SKIP)/STRIDE + 1)
    skel_images = np.zeros((img_num, SPATIAL_DIM*SPIXEL,TEMPORAL_DIM*SPIXEL,3), dtype=float)
    for img_ix in range(img_num):
        for frame_ix in xrange(TEMPORAL_DIM):
            current_frame = skel_norm[:,:,(img_ix*STRIDE + frame_ix*SKIP)]
            for order_ix in xrange(SPATIAL_DIM):
                skel_images[img_ix, order_ix*SPIXEL : (order_ix+1)*SPIXEL, frame_ix*SPIXEL : (frame_ix+1)*SPIXEL] = super_pixel(current_frame, order_ix+SPATIAL_DIM*seed_batch_idx)
        skel_images[img_ix,:,:,:] = cv2.normalize(skel_images[img_ix,:,:,:], skel_images[img_ix,:,:,:], 0, 1, cv2.NORM_MINMAX)
        skel_images[img_ix,:,:,:] = np.array(skel_images[img_ix,:,:,:] * 255, dtype = np.uint8)    
    return skel_images

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def main(args):
    TEMPORAL_DIM = int(args.image_w / 5)
    SPATIAL_DIM = int(args.image_h / 5)
    seed_batch_idx = args.seed_batch_idx

    newDataDir = os.path.join(save_dir, args.dataName)
    if not os.path.exists(newDataDir):
        os.makedirs(newDataDir)
    
    with tf.Graph().as_default():
        # Get a list of image paths and their labels
        action_list, label_list = get_image_paths_and_labels(dataPath=args.dataPath, s_ID=args.s_ID)
        
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #prelogits = tf.get_default_graph().get_tensor_by_name('InceptionResnetV1/Logits/Flatten/Reshape:0') # TF 1.2
            prelogits = tf.get_default_graph().get_tensor_by_name('InceptionResnetV1/Logits/Flatten/flatten/Reshape:0') # TF 1.4

            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
            for skel_file in tqdm(action_list):
                skel_norm = sio.loadmat(skel_file)
                skel_norm = skel_norm['sk']
                fm_num = skel_norm.shape[2]
                
                if fm_num < TEMPORAL_DIM:
                    skel_norm = skel_interpolate(skel_norm)
                    fm_num = skel_norm.shape[2]
                    if fm_num < TEMPORAL_DIM:
                        skel_norm = skel_interpolate(skel_norm)
                        fm_num = skel_norm.shape[2]
    
                ac_id = skel_file.split('/')[-1].split('.')[0]
                s_batch = 'nturgb+d_skel_s' + ac_id.split('C0')[0].split('S')[1]
                newActionDir = os.path.join(newDataDir, s_batch, ac_id)
                if not os.path.exists(newActionDir):	
                    os.makedirs(newActionDir)
    
                images = create_image_out_of_skeleton(skel_norm, TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx)
                images_prewhtien = np.zeros_like(images)
                for ix, one_img in enumerate(images):
                    one_img_prewhiten = prewhiten(one_img)
                    images_prewhtien[ix] = one_img_prewhiten
                
                feed_dict = { images_placeholder:images_prewhtien, phase_train_placeholder:False }
                if args.feature_type == 'embeddings':
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                if args.feature_type == 'prelogits':
                    emb_array = sess.run(prelogits, feed_dict=feed_dict)

                sio.savemat(os.path.join(newActionDir, ac_id+'.mat'), mdict={'featuresArr': emb_array})


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataPath', type=str,
        help='Directory where to write trained models and checkpoints.', default='/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/')
    parser.add_argument('--model', type=str,
        help='Directory where to write trained models and checkpoints.', default='/media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170925-103844/')
    parser.add_argument('--dataName', type=str,
        help='Directory where to write trained models and checkpoints.', default='COMPARE_Features_SuperPixel_5x5_32x32_S015_20170925-103844/')
    parser.add_argument('--image_h', type=int,
        help='Image size (height, width) in pixels.', default=180)
    parser.add_argument('--image_w', type=int,
        help='Image size (height, width) in pixels.', default=180)

    parser.add_argument('--feature_type', type=str, choices=['embeddings', 'prelogits'],
        help='Feature type.', default='embeddings')

    parser.add_argument('--s_ID', type=int, help='S batch.', default=None)

    # go to /media/jianl/TOSHIBA-EXT/Projects/Skeleton_Project/joints_correlation/NTU_scattering_degree_statics.py
    parser.add_argument('--seed_batch_idx', type=int, help='control the batch of random seed.', default=0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
