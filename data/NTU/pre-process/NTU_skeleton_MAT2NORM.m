clear
clc

skel_raw_dir = './nturgb+d_skeletons_SELECT_a50_a60_Two_Persons/';
skel_norm_dir = './nturgb+d_skeletons_NORMALIZE_a50_a60_Two_Persons/';

if ~exist(skel_norm_dir, 'dir')
    mkdir(skel_norm_dir);
end  

raw_Files = dir([skel_raw_dir '*.mat']);

%%
for s_idx = 1: size(raw_Files,1)
    fprintf('processing %s ...\n', raw_Files(s_idx).name);      
    skel_rawfile = [skel_raw_dir raw_Files(s_idx).name];
    load(skel_rawfile);
    skel_norm = [];
    for f_idx = 1:size(sk,3)
        skel = sk(:,:,f_idx);
        skel_new = NTU_skeleton_normalize(skel);
        skel_norm = cat(3, skel_norm, skel_new);
    end
    skel_normfile = [skel_norm_dir raw_Files(s_idx).name];
    sk = skel_norm;
    save(skel_normfile, 'sk');
end

