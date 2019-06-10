export PYTHONPATH=/media/jianl/TOSHIBA-EXT/Projects/facenet/src

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170922-005351/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170922-005351/ --image_h 120 --image_w 120

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170922-020628/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170922-020628/ --image_h 140 --image_w 140

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170922-032253/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170922-032253/ --image_h 160 --image_w 160

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170922-045050/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170922-045050/ --image_h 180 --image_w 180

#python facenet_feature_extractor_COMPARE_S015_Multi_Channel.py

#python facenet_feature_extractor_COMPARE_S015_Skel_Velo_Sandwich.py

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170925-055815/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170925-055815/ --image_h 160 --image_w 160

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170925-103844/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170925-103844/ --image_h 180 --image_w 180

#python facenet_feature_extractor_COMPARE_S015.py --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20170926-021203/ --dataName COMPARE_Features_SuperPixel_5x5_32x32_S015_20170926-021203/ --image_h 200 --image_w 200


python NTU_facenet_feature_extractor_COMPARE_S_All.py \
       --dataPath /media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/ \
       --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/ \
       --dataName S_All_Features_SuperPixel_5x5_36x36_Skeleton_XView_Dim_1792_20171003-155912/ \
       --image_h 180 --image_w 180 --feature_type prelogits --seed_batch_idx 0


python NTU_facenet_feature_extractor_COMPARE_S_All.py \
       --dataPath /media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_VELOCITY_a50_a60_Two_Persons/ \
       --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20171004-010154/ \
       --dataName S_All_Features_SuperPixel_5x5_36x36_Velocity_XView_Dim_1792_20171004-010154/ \
       --image_h 180 --image_w 180 --feature_type prelogits --seed_batch_idx 0


python NTU_facenet_feature_extractor_COMPARE_S_All.py \
       --dataPath /media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/ \
       --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20171005-214502/ \
       --dataName S_All_Features_SuperPixel_5x5_36x36_Skeleton_XSub_Dim_1792_20171005-214502/ \
       --image_h 180 --image_w 180 --feature_type prelogits --seed_batch_idx 2122


python facenet_feature_extractor_COMPARE_S015_Skel_Velo_Sandwich.py




### NUCLA
python NUCLA_facenet_feature_extractor_COMPARE_S_All.py \
       --dataPath /media/jianl/966022BE6022A4C9/Users/21884024/Jian/Datasets/NUCLA/NUCLA_skeletons_norm_Joints_25/ \
       --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/ntu_skeleton/compare_pseudo_image/S_All/20171003-155912/ \
       --dataName NUCLA_COMPARE_Features_SuperPixel_5x5_36x36_Skeleton/ \
       --image_h 180 --image_w 180 --feature_type prelogits --seed_batch_idx 0

# finetune for Sec01
python NUCLA_facenet_feature_extractor_COMPARE_S_All.py \
       --dataPath /media/jianl/966022BE6022A4C9/Users/21884024/Jian/Datasets/NUCLA/NUCLA_skeletons_norm_Joints_25/ \
       --model /media/jianl/TOSHIBA-EXT/Projects/facenet/models/nucla_skeleton/20171018-132606/ \
       --dataName NUCLA_COMPARE_Features_SuperPixel_5x5_36x36_Skeleton_finetune_Sec01/ \
       --image_h 180 --image_w 180 --feature_type prelogits --seed_batch_idx 0

