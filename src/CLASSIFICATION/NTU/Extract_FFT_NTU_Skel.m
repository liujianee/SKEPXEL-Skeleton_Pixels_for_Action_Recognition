clear all
clc

addpath('../../')

n_coeffs=[16 16 16];%% 3 levels temporal pyramid, 14/4=4 first fourier coefficients are used for each segment

temp1=[];

SKEL_Features = '../../../facenet_FEATURES/SKEL_Features/S_All_Features_SuperPixel_5x5_36x36_Skeleton_XView_Dim_1792_20171003-155912/';
SKEL_Descriptors = './Descriptors_FTP_SuperPixel_5x5_36_Skeleton/';

FEATURE_WIDTH = 1792;

s_list = dir([SKEL_Features 'ntu*']);

%%
for i = 1: length(s_list)

    s_dirs_batch = [s_list(i).name '/'];    
    
    skel_features_dir = [SKEL_Features s_dirs_batch];
    skel_descriptor_dir = [SKEL_Descriptors s_dirs_batch];
    
    list=dir([skel_features_dir 'S*']);
    
    final_features = zeros(length(list), FEATURE_WIDTH*28);
    labels = zeros(length(list), 5);
    
    if ~exist(skel_descriptor_dir, 'dir')
        mkdir(skel_descriptor_dir);
    end
    
    %%
    for i = 1 : length(list)
        fprintf('SKEL FTP Xview processing %s ...\n', list(i).name);
        
        load([skel_features_dir list(i).name '/' list(i).name '.mat']);%%features
        
        features = featuresArr'; %reshape(B, [75 fr_num]);
        
        temp1 = get_fourier_coeffs_pyramid(features, n_coeffs);
        temp1=temp1(:);
        
        temp1=temp1./norm(temp1);
        
        if size(temp1,1) ~= FEATURE_WIDTH*28 %% e.g. S015C001P008R002A040
            temp1 = [temp1; zeros(FEATURE_WIDTH*28 - size(temp1,1), 1)];
        end
        
        final_features(i,:) = temp1(:)';
        
        clear('temp1','features');
        
        s_idx = str2num(list(i).name(1,2:4));
        view = str2num(list(i).name(1,6:8));
        subject = str2num(list(i).name(1,10:12));
        r_idx = str2num(list(i).name(1,14:16));
        action = str2num(list(i).name(1,18:20));
    
        labels(i,1:5)=[s_idx view subject r_idx action];
    end
    fprintf('all actions done, start saving ...\n');
    %%
    save([skel_descriptor_dir 'Labels_114688.mat'],'labels');
    %save([s_dirs_batch 'final_features_114688.mat'],'final_features');
    save([skel_descriptor_dir 'final_features_114688.mat'],'final_features','-v7.3');
    
    fprintf('saving complete...\n');

end


