#python src/train_softmax_NTU_Skeleton_COMPARE_S_All_Multi_Channel.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_channel 6

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All_Skel_Velo_Sandwich.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_channel 6

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 160 --image_w 160

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 1 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 200 --image_w 200

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All_float_type_image.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 1 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All_Skel_Velo_Mosaic.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --embedding_size 256

#python src/train_softmax_NTU_Skeleton_NetVLAD.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 220 --image_w 220 --batch_size 50

#python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 240 --image_w 240 --batch_size 50

#python src/train_softmax_NTU_Skeleton_use_Pretrained_Model_NetVLAD.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --pretrained_model models/facenet/20170512-110547/model-20170512-110547.ckpt-250000

python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180

python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --dataPath /media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_VELOCITY_a50_a60_Two_Persons/

python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --x_type XSub --seed_batch_idx 2122

python src/train_softmax_NTU_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --dataPath /media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_VELOCITY_a50_a60_Two_Persons/ --x_type XSub

## skel_velo_sandwich
python src/train_softmax_NTU_Skeleton_COMPARE_S_All_Skel_Velo_Sandwich.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180

python src/train_softmax_NTU_Skeleton_COMPARE_S_All_Skel_Velo_Sandwich.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 10 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --x_type XSub




## UWA3D
python src/train_softmax_UWA3D_Skeleton_COMPARE_S_All.py --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 1 --keep_probability 0.8 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --image_h 180 --image_w 180 --x_type 0 --seed_batch_idx 257 --epoch_size 500


python src/train_softmax_UWA3D_Skeleton.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/uwa3d_skeleton/SuperPixel_4x4_36/cross_view/Sec0/train/ \
       --test_data_dir datasets/uwa3d_skeleton/SuperPixel_4x4_36/cross_view/Sec0/test/ \
       --image_size 144


python src/train_softmax_UWA3D_Skeleton.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/uwa3d_skeleton/SuperPixel_5x5_36/cross_view/Sec0/train/ \
       --test_data_dir datasets/uwa3d_skeleton/SuperPixel_5x5_36/cross_view/Sec0/test/ \
       --image_size 180


python src/train_softmax_UWA3D_Skeleton_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 5 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/uwa3d_skeleton/SuperPixel_5x5_36/cross_view/Sec0/train/ \
       --test_data_dir datasets/uwa3d_skeleton/SuperPixel_5x5_36/cross_view/Sec0/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/model-20171003-155912.ckpt-10000 \
       --image_size 180


## NUCLA
## Sec0
python src/train_softmax_NUCLA_Skeleton_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec0/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec0/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/model-20171003-155912.ckpt-10000 \
       --image_size 180

python src/train_softmax_NUCLA_Velocity_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec0/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec0/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171004-010154/model-20171004-010154.ckpt-10000 \
       --image_size 180



## Sec1
python src/train_softmax_NUCLA_Skeleton_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec1/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec1/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/model-20171003-155912.ckpt-10000 \
       --image_size 180

python src/train_softmax_NUCLA_Velocity_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec1/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec1/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171004-010154/model-20171004-010154.ckpt-10000 \
       --image_size 180



## Sec2
python src/train_softmax_NUCLA_Skeleton_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec2/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36/cross_view/Sec2/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/model-20171003-155912.ckpt-10000 \
       --image_size 180

python src/train_softmax_NUCLA_Velocity_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec2/train/ \
       --test_data_dir datasets/nucla_skeleton/SuperPixel_5x5_36_VELOCITY/cross_view/Sec2/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171004-010154/model-20171004-010154.ckpt-10000 \
       --image_size 180



## skeleton_velocity
## Sec0
python src/train_softmax_NUCLA_Skeleton_Skel_Velo_Sandwich_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --dataPath /media/jianl/966022BE6022A4C9/Users/21884024/Jian/Datasets/NUCLA/NUCLA_skeletons_SELECT_VELOCITY_COMBINED/ \
       --pretrained_model models/ntu_skeleton/skel_velo_sandwich/S_All/20171008-201541/model-20171008-201541.ckpt-7000
       --image_size 180 --x_type 0 --epoch_size 5


## UTH

python src/train_softmax_UTH_Skeleton_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/uth_skeleton/SuperPixel_5x5_36/cross_sub/train/ \
       --test_data_dir datasets/uth_skeleton/SuperPixel_5x5_36/cross_sub/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/model-20171003-155912.ckpt-10000 \
       --image_size 180

python src/train_softmax_UTH_Velocity_finetune.py \
       --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 2 --keep_probability 0.8 \
       --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
       --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
       --data_dir datasets/uth_skeleton/SuperPixel_5x5_36_VELOCITY/cross_sub/train/ \
       --test_data_dir datasets/uth_skeleton/SuperPixel_5x5_36_VELOCITY/cross_sub/test/ \
       --pretrained_model models/ntu_skeleton/compare_pseudo_image/S_All/20171004-010154/model-20171004-010154.ckpt-10000 \
       --image_size 180


