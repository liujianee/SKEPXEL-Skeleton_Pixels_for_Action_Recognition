clear
close all

data_skel = './nturgb+d_skeletons_SELECT_a50_a60_Two_Persons/';
data_velo = './nturgb+d_skeletons_VELOCITY_a50_a60_Two_Persons/';

ac_files_skel = dir([data_skel '*.mat']);
ac_files_velo = dir([data_velo '*.mat']);

saveDir = './nturgb+d_skeletons_SELECT_VELOCITY_COMBINED/';

if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end   

%%
for ac = 1:length(ac_files_skel)
    fprintf('processing %s ...\n', ac_files_skel(ac).name);
    C = strsplit(ac_files_skel(ac).name, 'A');
    CC = strsplit(C{2}, '.');
    acion_ID = str2num(CC{1});    
    
    skelFile_skel = [data_skel ac_files_skel(ac).name];
    skelFile_velo = [data_velo ac_files_velo(ac).name];    
    
    load(skelFile_skel);    sk_skel = sk;
    load(skelFile_velo);    sk_velo = sk;
    
    fm_num_skel = size(sk_skel, 3);
    fm_num_velo = size(sk_velo, 3);    
    
    fm_num_diff = fm_num_skel - fm_num_velo; 
    
    sk_skel(:,:,1:fm_num_diff) = [];
    
    sk = cat(2, sk_skel, sk_velo);
    saveFile = [saveDir ac_files_skel(ac).name];
    save(saveFile, 'sk');   
end
