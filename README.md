# Skepxels: Spatio-temporal Image Representation of Human Skeleton Joints for Action Recognition
This repository contains the implementation of ["Skepxels: Spatio-temporal Image Representation of Human Skeleton Joints for Action Recognition"](https://arxiv.org/abs/1711.05941)

<img src="https://github.com/liujianee/Skepxel_GitHub/blob/master/assets/skeleton_pixel.png" width="50%">

<img src="https://github.com/liujianee/Skepxel_GitHub/blob/master/assets/image_generation.png" width="50%">

<img src="https://github.com/liujianee/Skepxel_GitHub/blob/master/assets/target_images.png" width="50%">

## Environment
- Python 2.7
- Tensorflow 1.9.0
- Matlab


## Usage

### ENV SETUP
1. If you are using conda, you can create a virtual environment by `conda env create -f environment.yml`

2. Download and config [facenet](https://github.com/davidsandberg/facenet) repository.  

3. Modify pathes in below scripts to fit your own system.

### DATA PRCOSSING

1. Download [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionRecognition.asp) action dataset. Only skeleton data is used here.

2. Preproess the skeleton data with [scripts](https://github.com/liujianee/Skepxel_GitHub/tree/master/data/NTU/pre-process)


### TESTING

1. Download trained [models](https://github.com/liujianee/Skepxel_GitHub/tree/master/models/ntu_skeleton/compare_pseudo_image/S_ALL/20171003-155912)

2. Extract features of skepxel images with [scripts](https://github.com/liujianee/Skepxel_GitHub/tree/master/src/FEATURES).

3. Apply [FTP](https://github.com/liujianee/Skepxel_GitHub/blob/master/src/CLASSIFICATION/NTU/Extract_FFT_NTU_Skel.m) on extracted features and [split](https://github.com/liujianee/Skepxel_GitHub/blob/master/src/CLASSIFICATION/NTU/NTU_datasetGen.m) the results according to testing protocols.

4. Run [classification](https://github.com/liujianee/Skepxel_GitHub/blob/master/src/CLASSIFICATION/NTU/NTU_liblinear_SVM_XView_multibatch.py).


### TRAINING

1. Generate skexepl images for with [scripts](https://github.com/liujianee/Skepxel_GitHub/tree/master/data/NTU/skepxel_images).

2. Place [script](https://github.com/liujianee/Skepxel_GitHub/blob/master/src/facenet/src/train_softmax_NTU_Skeleton_COMPARE_S_All.py) to `facenet/src`. 

3. Config and Run [TRAINING script](https://github.com/liujianee/Skepxel_GitHub/blob/master/src/Train_script.sh).


## References
- [facenet](https://github.com/davidsandberg/facenet)

