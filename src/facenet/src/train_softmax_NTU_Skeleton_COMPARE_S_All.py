"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import scipy.io as sio
import time
import threading

##
#TEMPORAL_DIM = 32
#SPATIAL_DIM = 32
#TEMPORAL_DIM = 36
#SPATIAL_DIM = 36

SPIXEL = 5
STRIDE = 1	# decide how many pseudo images to be created
SKIP = 1	# decide how dense/sparse the skeleton frames are sampled, to build one pseudo image


# normalized skeleton dataset
#dataPath = '/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/'

train_subject_id = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
test_subject_id = [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40]

train_view_id = [2,3]
test_view_id = [1]


def get_image_paths_and_labels(dataPath=None, s_ID=None, is_training=True, x_type='XView'):
    image_list = []
    label_list = []
    ac_Dirs = os.listdir(dataPath)
    if s_ID == 15:
        ac_Dirs_batch = ac_Dirs[46606:49346]
    else:
        ac_Dirs_batch = ac_Dirs	# TODO
    for one_ac in ac_Dirs_batch:
        view_ID = int(one_ac.split('P')[0].split('C')[1])
        sub_ID = int(one_ac.split('R')[0].split('P')[1])
        action_ID = int(one_ac.split('.')[0].split('A')[1]) - 1
        ac_matfile = dataPath + one_ac
        if is_training:
            if (x_type=='XView') & (view_ID in train_view_id):
                image_list.append(ac_matfile)
                label_list.append(action_ID)
            if (x_type=='XSub') & (sub_ID in train_subject_id):
                image_list.append(ac_matfile)
                label_list.append(action_ID)
        else:
            if (x_type=='XView') & (view_ID in test_view_id):
                image_list.append(ac_matfile)
                label_list.append(action_ID)
            if (x_type=='XSub') & (sub_ID in test_subject_id):
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


def create_image_out_of_skeleton(skel_norm, img_ix, TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx):
    skel_arr = np.zeros((SPATIAL_DIM*SPIXEL,TEMPORAL_DIM*SPIXEL,3), dtype=float)
    for frame_ix in xrange(TEMPORAL_DIM):
      current_frame = skel_norm[:,:,(img_ix*STRIDE + frame_ix*SKIP)]
      for order_ix in xrange(SPATIAL_DIM):
        skel_arr[order_ix*SPIXEL : (order_ix+1)*SPIXEL, frame_ix*SPIXEL : (frame_ix+1)*SPIXEL] = super_pixel(current_frame, order_ix+SPATIAL_DIM*seed_batch_idx)
    skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
    skel_img = np.array(skel_img * 255, dtype = np.uint8)    
    return skel_img


def enqueue(sess, index_dequeue_op, enqueue_op, action_list, label_list, queue_input_data, queue_input_label, TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx):
    while True:
        print("starting to write into queue")
        index_epoch = sess.run(index_dequeue_op)
        label_epoch = np.array(label_list)[index_epoch]
        action_epoch = np.array(action_list)[index_epoch]
        #print("try to enqueue ", index_epoch[0:10])
        curr_data = []
        curr_label = []
        for one_lab, ac_matfile in zip(label_epoch, action_epoch):
            mat = sio.loadmat(ac_matfile)
            skel_norm = mat['sk']
            fm_num = skel_norm.shape[2]
            if fm_num < TEMPORAL_DIM:
                skel_norm = skel_interpolate(skel_norm)
                fm_num = skel_norm.shape[2]
                if fm_num < TEMPORAL_DIM:
                    skel_norm = skel_interpolate(skel_norm)
                    fm_num = skel_norm.shape[2]
            # the number of pseudo images created from this skeleton
            img_num = int((fm_num - TEMPORAL_DIM*SKIP)/STRIDE + 1)
            random.seed(time.time())
            #img_ix = random.choice(xrange(img_num))
            img_ix = int(random.choice(xrange(img_num))/5) * 5	# sample images to train
            skel_img = create_image_out_of_skeleton(skel_norm, img_ix, TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx)
            curr_data.append(skel_img)
            curr_label.append(one_lab) #####################################
        curr_data_arr = np.array(curr_data)
        curr_label_arr = np.expand_dims(np.array(curr_label), 1)
        sess.run(enqueue_op, feed_dict={queue_input_data: curr_data_arr,
                                        queue_input_label: curr_label_arr})
        print("added to the queue")
    print("finished enqueueing")


def main(args):
    TEMPORAL_DIM = int(args.image_w / 5)
    SPATIAL_DIM = int(args.image_h / 5)
    seed_batch_idx = args.seed_batch_idx

    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        action_list, label_list = get_image_paths_and_labels(dataPath=args.dataPath, s_ID=None, is_training=True, x_type=args.x_type)
        assert len(action_list)>0, 'The dataset should not be empty'
        
        ## create input pipeline
        queue_input_data = tf.placeholder(tf.float32, shape=[None, args.image_h, args.image_w, 3])
        queue_input_label = tf.placeholder(tf.int64, shape=[None, 1])

        queue = tf.FIFOQueue(capacity=20000, dtypes=[tf.float32, tf.int64], shapes=[[args.image_h, args.image_w, 3], [1]])
        enqueue_op = queue.enqueue_many([queue_input_data, queue_input_label])

        dequeue_data, dequeue_lab = queue.dequeue()
        dequeue_image = tf.image.per_image_standardization(dequeue_data)

        # Create a queue that produces indices into the action_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)

        #index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*100, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        
        #image_batch, label_batch = tf.train.batch([dequeue_image, dequeue_lab], batch_size=args.batch_size, capacity=100)
        image_batch, label_batch = tf.train.batch([dequeue_image, dequeue_lab], batch_size=batch_size_placeholder, capacity=1000)

        label_batch = tf.squeeze(label_batch)

        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        tf.summary.image("input_image", image_batch, max_outputs=4)

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(action_list))
        
        print('Building training graph')
        
        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        

        # Calculate batch accuracy
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), label_batch)
        batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='batch_accuracy')
        tf.summary.scalar('batch_accuracy', batch_accuracy)


        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one 0batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        enqueue_thread = threading.Thread(target=enqueue, args=[sess, index_dequeue_op, enqueue_op, action_list, label_list, 
                                                                queue_input_data, queue_input_label, 
                                                                TEMPORAL_DIM, SPATIAL_DIM, seed_batch_idx])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, #image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                if args.lfw_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
                        embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)

        sess.run(queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        sess.close()

    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, #image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

#    index_epoch = sess.run(index_dequeue_op)
#    label_epoch = np.array(label_list)[index_epoch]
#    image_epoch = np.array(image_list)[index_epoch]
#    
#    # Enqueue one epoch of image paths and labels
#    labels_array = np.expand_dims(np.array(label_epoch),1)
#    image_paths_array = np.expand_dims(np.array(image_epoch),1)
#    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    
    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame)*2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb
        
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/ntu_skeleton/compare_pseudo_image/S_All/')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/ntu_skeleton/compare_pseudo_image/S_All/')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        #default='datasets/ntu_skeleton/SuperPixel_5x5_32_a50_a60_Two_Persons/cross_view/train/')
        default='datasets/ntu_skeleton/SuperPixel_5x5_32_skel_fully_normed_no_augmt_S015/cross_view/train/')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_h', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_w', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--x_type', type=str, choices=['XView', 'XSub'],
        help='The optimization algorithm to use', default='XView')
    # go to /media/jianl/TOSHIBA-EXT/Projects/Skeleton_Project/joints_correlation/NTU_scattering_degree_statics.py
    parser.add_argument('--seed_batch_idx', type=int, choices=[0, 2122, 1866],
        help='control the batch of random seed.', default=0)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)

    parser.add_argument('--dataPath', type=str,
        help='Directory where to write trained models and checkpoints.', default='/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
