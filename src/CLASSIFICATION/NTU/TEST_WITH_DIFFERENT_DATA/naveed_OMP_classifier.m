function [  ] = experiments_PR_Jian(  )
%% Final experiments for the results in the first manuscript submission to PR
%clear
%DataBase = processUWA('342'); % options 124, 142, 234
%DataBase = processNUCLA('231'); % options 123, 132, 231
%DataBase = processMSR();


% for UWA 0.1, 0.75 is ok
% for NUCLA 0.01, 0.4 give good results
% for MSR 0.
param.lambda = 0.01;
param.lambda1 = 0.35;

param.sparsity = 50;
SparseDenseCRCHere(DataBase, param);

end


function [  ] = SparseDenseCRCHere( DataBase, param )
train = DataBase.training_feats;
Y = DataBase.testing_feats;
W = DataBase.H_train;
test_label = DataBase.H_test;
% 
% train = train(:, 1:30000);
% W = W(:, 1:30000);

tic
Phi = train;
P = (Phi' * Phi + (param.lambda* eye(size(Phi,2))))\Phi' ;
A_check = P * Y;
%n_A_check = sqrt(sum(A_check.^2));
G = Phi'*Phi;
addpath('/media/jianl/disk3/Jian/Datasets/NTU/CLASSIFICATION/Naveed_For_PR_NTU_Exp/ompbox10')
A_hat = omp(Phi'*Y,G,param.sparsity);
%n_A_hat = sqrt(sum(A_hat.^2));

%A_check1 = A_check./(repmat(n_A_check+n_A_hat , size(A_check, 1), 1));
%A_hat1 = A_hat./(repmat(n_A_check+n_A_hat , size(A_hat, 1), 1));

Score = W *(  (   (1-param.lambda1)*(normc(A_check))   ) + (   (param.lambda1)*(normc(A_hat))   )  );


%% Classification results:
errnum = 0;
Nt = size(Y,2);
for featureid=1:size(Y,2)
    score_est =  Score(:,featureid);
    score_gt = test_label(:,featureid);
    [maxv_est, maxind_est] = max(score_est);  
    [maxv_gt, maxind_gt] = max(score_gt);
    if(maxind_est~=maxind_gt)
        errnum = errnum + 1;
    end
end
accuracy_SDCRC = (Nt-errnum)/Nt *100;
time_SDCRC = toc;
st = ['Classification accuracy is ',  num2str(accuracy_SDCRC), '% in ' , num2str(time_SDCRC), ' seconds'];
disp (st)

clear A_check A_hat G P Phi Score W Y train trl_onehot tel_onehot test_label;

end

function [DataBase] = processUWA(X)
%path = 'C:\Users\NAVEED AKHTAR\Documents\MATLAB\Jian PR\TO_NAVEED\UWA\features_Sec_Comb\Sec_';
path = 'E:\Action\UWA3D\Sec_';
te = load([path X '\testData']);
tr = load([path X '\trainData']);
tel = load([path X '\testLab']);
trl = load([path X '\trainLab']);

DataBase.testing_feats = (te.testData');
DataBase.training_feats = normc(tr.trainData');
DataBase.H_train = trl.trainLab';
DataBase.H_test = tel.testLab';
end
function [DataBase] = processNUCLA(X)
path = 'C:\Users\NAVEED AKHTAR\Documents\MATLAB\Jian PR\TO_NAVEED\NUCLA\features_Sec_Comb\Sec_';
te = load([path X '\testData']);
tr = load([path X '\trainData']);
tel = load([path X '\testLab']);
trl = load([path X '\trainLab']);

DataBase.testing_feats = (te.testData');
DataBase.training_feats = normc(tr.trainData');
DataBase.H_train = trl.trainLab';
DataBase.H_test = tel.testLab';
end

function [DataBase] = processMSR()
path = 'C:\Users\NAVEED AKHTAR\Documents\MATLAB\Jian PR\TO_NAVEED\MSR';
%path = 'C:\Users\NAVEED AKHTAR\Dropbox\MSR\features_FTP_4';
te = load([path '\testData']);
tr = load([path '\trainData']);
tel = load([path '\testLab']);
trl = load([path '\trainLab']);

DataBase.testing_feats = (te.testData');
DataBase.training_feats = normc(tr.trainData');
DataBase.H_train = trl.trainLab';
DataBase.H_test = tel.testLab';
end

function [DataBase] = processNTU_HPM_NKTM_XSub()

hpm_work_dir = '../HPM_Descriptors_GoogLeNet_GAN/HPM_Output_Norm/';
nktm_work_dir = '/media/jianl/MULTIPIE/JIANLIU/Datasets/NTU/NKTM/CLASSIFICATION/NKTM_Descriptors/NKTM_Output/'

hpm_s_list = dir([hpm_work_dir 'ntu*']);
nktm_s_list = dir([nktm_work_dir 'ntu*']);

te = [];
tel = [];
tr = [];
trl = [];

te = zeros( 54210, 34672);
tr = zeros( 54210, 34672);

te_start_idx = 1;
tr_start_idx = 1;

for i = 1 : length(hpm_s_list)
    fprintf('processing %s ...\n', hpm_s_list(i).name);
    
    hpm_trainData_mat = [hpm_work_dir hpm_s_list(i).name '/trainData_sub.mat'];
    hpm_trainLab_mat = [hpm_work_dir hpm_s_list(i).name '/trainLab_sub.mat'];
    hpm_testData_mat = [hpm_work_dir hpm_s_list(i).name '/testData_sub.mat'];
    hpm_testLab_mat = [hpm_work_dir hpm_s_list(i).name '/testLab_sub.mat'];
    load(hpm_trainData_mat); hpm_trainData = trainData; clear trainData;
    load(hpm_trainLab_mat); hpm_trainLab = trainLab; clear trainLab;
    load(hpm_testData_mat); hpm_testData = testData; clear testData;
    load(hpm_testLab_mat); hpm_testLab = testLab; clear testLab;
    
    nktm_trainData_mat = [nktm_work_dir nktm_s_list(i).name '/trainData_sub.mat'];
    nktm_trainLab_mat = [nktm_work_dir nktm_s_list(i).name '/trainLab_sub.mat'];
    nktm_testData_mat = [nktm_work_dir nktm_s_list(i).name '/testData_sub.mat'];
    nktm_testLab_mat = [nktm_work_dir nktm_s_list(i).name '/testLab_sub.mat'];   
    load(nktm_trainData_mat); nktm_trainData = trainData; clear trainData;
    load(nktm_trainLab_mat); nktm_trainLab = trainLab; clear trainLab;
    load(nktm_testData_mat); nktm_testData = testData; clear testData;
    load(nktm_testLab_mat); nktm_testLab = testLab; clear testLab;
            
    if (isequal(hpm_trainLab, nktm_trainLab) && isequal(hpm_testLab, nktm_testLab))
%         tr = [tr; [hpm_trainData, nktm_trainData]];
        tr(tr_start_idx : tr_start_idx+size(hpm_trainData,1)-1 , :) = [hpm_trainData, nktm_trainData];
        trl = [trl; hpm_trainLab(:,5)];
%         te = [te; [hpm_testData, nktm_testData]];
        te(te_start_idx : te_start_idx+size(hpm_testData,1)-1 , :) = [hpm_testData, nktm_testData];
        tel = [tel; hpm_testLab(:,5)];
    else
        fprintf('ERROR! mismatch: %s \n', hpm_s_list(i).name);
    end
    
    tr_start_idx = tr_start_idx + size(hpm_trainData,1);
    te_start_idx = te_start_idx + size(hpm_testData,1);
    
end

tr(tr_start_idx:end, :) = [];
te(te_start_idx:end, :) = [];

n_classes =60;

trl_onehot = zeros( size( trl, 1 ), n_classes);
for i = 1:n_classes
    rows = trl == i;
    trl_onehot( rows, i ) = 1;
end

tel_onehot = zeros( size( tel, 1 ), n_classes);
for i = 1:n_classes
    rows = tel == i;
    tel_onehot( rows, i ) = 1;
end

DataBase.testing_feats = te';
DataBase.training_feats = normc(tr');
DataBase.H_train = trl_onehot';
DataBase.H_test = tel_onehot';
end

function [DataBase] = processNTU_HPM_NKTM_XView()

hpm_work_dir = '../HPM_Descriptors_GoogLeNet_GAN/HPM_Output_Norm/';
nktm_work_dir = '/media/jianl/MULTIPIE/JIANLIU/Datasets/NTU/NKTM/CLASSIFICATION/NKTM_Descriptors/NKTM_Output/'

hpm_s_list = dir([hpm_work_dir 'ntu*']);
nktm_s_list = dir([nktm_work_dir 'ntu*']);

te = [];
tel = [];
tr = [];
trl = [];

te = zeros( 54210, 34672);
tr = zeros( 54210, 34672);

te_start_idx = 1;
tr_start_idx = 1;

for i = 1 : length(hpm_s_list)
    fprintf('processing %s ...\n', hpm_s_list(i).name);
    
    hpm_trainData_mat = [hpm_work_dir hpm_s_list(i).name '/trainData_view.mat'];
    hpm_trainLab_mat = [hpm_work_dir hpm_s_list(i).name '/trainLab_view.mat'];
    hpm_testData_mat = [hpm_work_dir hpm_s_list(i).name '/testData_view.mat'];
    hpm_testLab_mat = [hpm_work_dir hpm_s_list(i).name '/testLab_view.mat'];
    load(hpm_trainData_mat); hpm_trainData = trainData; clear trainData;
    load(hpm_trainLab_mat); hpm_trainLab = trainLab; clear trainLab;
    load(hpm_testData_mat); hpm_testData = testData; clear testData;
    load(hpm_testLab_mat); hpm_testLab = testLab; clear testLab;
    
    nktm_trainData_mat = [nktm_work_dir nktm_s_list(i).name '/trainData_view.mat'];
    nktm_trainLab_mat = [nktm_work_dir nktm_s_list(i).name '/trainLab_view.mat'];
    nktm_testData_mat = [nktm_work_dir nktm_s_list(i).name '/testData_view.mat'];
    nktm_testLab_mat = [nktm_work_dir nktm_s_list(i).name '/testLab_view.mat'];   
    load(nktm_trainData_mat); nktm_trainData = trainData; clear trainData;
    load(nktm_trainLab_mat); nktm_trainLab = trainLab; clear trainLab;
    load(nktm_testData_mat); nktm_testData = testData; clear testData;
    load(nktm_testLab_mat); nktm_testLab = testLab; clear testLab;
            
    if (isequal(hpm_trainLab, nktm_trainLab) && isequal(hpm_testLab, nktm_testLab))
%         tr = [tr; [hpm_trainData, nktm_trainData]];
        tr(tr_start_idx : tr_start_idx+size(hpm_trainData,1)-1 , :) = [hpm_trainData, nktm_trainData];
        trl = [trl; hpm_trainLab(:,5)];
%         te = [te; [hpm_testData, nktm_testData]];
        te(te_start_idx : te_start_idx+size(hpm_testData,1)-1 , :) = [hpm_testData, nktm_testData];
        tel = [tel; hpm_testLab(:,5)];
    else
        fprintf('ERROR! mismatch: %s \n', hpm_s_list(i).name);
    end
    
    tr_start_idx = tr_start_idx + size(hpm_trainData,1);
    te_start_idx = te_start_idx + size(hpm_testData,1);
    
end

tr(tr_start_idx:end, :) = [];
te(te_start_idx:end, :) = [];

n_classes =60;

trl_onehot = zeros( size( trl, 1 ), n_classes);
for i = 1:n_classes
    rows = trl == i;
    trl_onehot( rows, i ) = 1;
end

tel_onehot = zeros( size( tel, 1 ), n_classes);
for i = 1:n_classes
    rows = tel == i;
    tel_onehot( rows, i ) = 1;
end

DataBase.testing_feats = te';
DataBase.training_feats = normc(tr');
DataBase.H_train = trl_onehot';
DataBase.H_test = tel_onehot';
end

function [DataBase] = processNTU_HPM_NKTM_XSub_Scaled()

hpm_work_dir = '../HPM_Descriptors_GoogLeNet_GAN/HPM_Output/';
nktm_work_dir = '/media/jianl/MULTIPIE/JIANLIU/Datasets/NTU/NKTM/CLASSIFICATION/NKTM_Descriptors/NKTM_Output/'

hpm_s_list = dir([hpm_work_dir 'ntu*']);
nktm_s_list = dir([nktm_work_dir 'ntu*']);

te = [];
tel = [];
tr = [];
trl = [];

te = zeros( 54210, 34672);
tr = zeros( 54210, 34672);

te_start_idx = 1;
tr_start_idx = 1;

for i = 1 : length(hpm_s_list)
    fprintf('processing %s ...\n', hpm_s_list(i).name);
    
    hpm_trainData_mat = [hpm_work_dir hpm_s_list(i).name '/trainData_sub.mat'];
    hpm_trainLab_mat = [hpm_work_dir hpm_s_list(i).name '/trainLab_sub.mat'];
    hpm_testData_mat = [hpm_work_dir hpm_s_list(i).name '/testData_sub.mat'];
    hpm_testLab_mat = [hpm_work_dir hpm_s_list(i).name '/testLab_sub.mat'];
    load(hpm_trainData_mat); hpm_trainData = trainData; clear trainData;
    load(hpm_trainLab_mat); hpm_trainLab = trainLab; clear trainLab;
    load(hpm_testData_mat); hpm_testData = testData; clear testData;
    load(hpm_testLab_mat); hpm_testLab = testLab; clear testLab;
    
    nktm_trainData_mat = [nktm_work_dir nktm_s_list(i).name '/trainData_sub.mat'];
    nktm_trainLab_mat = [nktm_work_dir nktm_s_list(i).name '/trainLab_sub.mat'];
    nktm_testData_mat = [nktm_work_dir nktm_s_list(i).name '/testData_sub.mat'];
    nktm_testLab_mat = [nktm_work_dir nktm_s_list(i).name '/testLab_sub.mat'];   
    load(nktm_trainData_mat); nktm_trainData = trainData; clear trainData;
    load(nktm_trainLab_mat); nktm_trainLab = trainLab; clear trainLab;
    load(nktm_testData_mat); nktm_testData = testData; clear testData;
    load(nktm_testLab_mat); nktm_testLab = testLab; clear testLab;
            
    if (isequal(hpm_trainLab, nktm_trainLab) && isequal(hpm_testLab, nktm_testLab))
%         tr = [tr; [hpm_trainData, nktm_trainData]];
        tr(tr_start_idx : tr_start_idx+size(hpm_trainData,1)-1 , :) = [scaleDescs(hpm_trainData), scaleDescs(nktm_trainData)];
        trl = [trl; hpm_trainLab(:,5)];
%         te = [te; [hpm_testData, nktm_testData]];
        te(te_start_idx : te_start_idx+size(hpm_testData,1)-1 , :) = [scaleDescs(hpm_testData), scaleDescs(nktm_testData)];
        tel = [tel; hpm_testLab(:,5)];
    else
        fprintf('ERROR! mismatch: %s \n', hpm_s_list(i).name);
    end
    
    tr_start_idx = tr_start_idx + size(hpm_trainData,1);
    te_start_idx = te_start_idx + size(hpm_testData,1);
    
end

tr(tr_start_idx:end, :) = [];
te(te_start_idx:end, :) = [];

n_classes =60;

trl_onehot = zeros( size( trl, 1 ), n_classes);
for i = 1:n_classes
    rows = trl == i;
    trl_onehot( rows, i ) = 1;
end

tel_onehot = zeros( size( tel, 1 ), n_classes);
for i = 1:n_classes
    rows = tel == i;
    tel_onehot( rows, i ) = 1;
end

DataBase.testing_feats = te';
DataBase.training_feats = normc(tr');
DataBase.H_train = trl_onehot';
DataBase.H_test = tel_onehot';

save('DataBase_XSub_Scaled','DataBase', '-v7.3');
end

function [DataBase] = processNTU_HPM_NKTM_XView_Scaled()

hpm_work_dir = '../HPM_Descriptors_GoogLeNet_GAN/HPM_Output/';
nktm_work_dir = '/media/jianl/MULTIPIE/JIANLIU/Datasets/NTU/NKTM/CLASSIFICATION/NKTM_Descriptors/NKTM_Output/';

hpm_s_list = dir([hpm_work_dir 'ntu*']);
nktm_s_list = dir([nktm_work_dir 'ntu*']);

te = [];
tel = [];
tr = [];
trl = [];

te = zeros( 54210, 34672);
tr = zeros( 54210, 34672);

te_start_idx = 1;
tr_start_idx = 1;

for i = 1 : length(hpm_s_list)
    fprintf('processing %s ...\n', hpm_s_list(i).name);
    
    hpm_trainData_mat = [hpm_work_dir hpm_s_list(i).name '/trainData_view.mat'];
    hpm_trainLab_mat = [hpm_work_dir hpm_s_list(i).name '/trainLab_view.mat'];
    hpm_testData_mat = [hpm_work_dir hpm_s_list(i).name '/testData_view.mat'];
    hpm_testLab_mat = [hpm_work_dir hpm_s_list(i).name '/testLab_view.mat'];
    load(hpm_trainData_mat); hpm_trainData = trainData; clear trainData;
    load(hpm_trainLab_mat); hpm_trainLab = trainLab; clear trainLab;
    load(hpm_testData_mat); hpm_testData = testData; clear testData;
    load(hpm_testLab_mat); hpm_testLab = testLab; clear testLab;
    
    nktm_trainData_mat = [nktm_work_dir nktm_s_list(i).name '/trainData_view.mat'];
    nktm_trainLab_mat = [nktm_work_dir nktm_s_list(i).name '/trainLab_view.mat'];
    nktm_testData_mat = [nktm_work_dir nktm_s_list(i).name '/testData_view.mat'];
    nktm_testLab_mat = [nktm_work_dir nktm_s_list(i).name '/testLab_view.mat'];   
    load(nktm_trainData_mat); nktm_trainData = trainData; clear trainData;
    load(nktm_trainLab_mat); nktm_trainLab = trainLab; clear trainLab;
    load(nktm_testData_mat); nktm_testData = testData; clear testData;
    load(nktm_testLab_mat); nktm_testLab = testLab; clear testLab;
            
    if (isequal(hpm_trainLab, nktm_trainLab) && isequal(hpm_testLab, nktm_testLab))
%         tr = [tr; [hpm_trainData, nktm_trainData]];
        tr(tr_start_idx : tr_start_idx+size(hpm_trainData,1)-1 , :) = [scaleDescs(hpm_trainData), scaleDescs(nktm_trainData)];
        trl = [trl; hpm_trainLab(:,5)];
%         te = [te; [hpm_testData, nktm_testData]];
        te(te_start_idx : te_start_idx+size(hpm_testData,1)-1 , :) = [scaleDescs(hpm_testData), scaleDescs(nktm_testData)];
        tel = [tel; hpm_testLab(:,5)];
    else
        fprintf('ERROR! mismatch: %s \n', hpm_s_list(i).name);
    end
    
    tr_start_idx = tr_start_idx + size(hpm_trainData,1);
    te_start_idx = te_start_idx + size(hpm_testData,1);
    
end

tr(tr_start_idx:end, :) = [];
te(te_start_idx:end, :) = [];

n_classes =60;

trl_onehot = zeros( size( trl, 1 ), n_classes);
for i = 1:n_classes
    rows = trl == i;
    trl_onehot( rows, i ) = 1;
end

tel_onehot = zeros( size( tel, 1 ), n_classes);
for i = 1:n_classes
    rows = tel == i;
    tel_onehot( rows, i ) = 1;
end

DataBase.testing_feats = te';
DataBase.training_feats = normc(tr');
DataBase.H_train = trl_onehot';
DataBase.H_test = tel_onehot';

save('DataBase_XView_Scaled','DataBase', '-v7.3');
end

function [DataBase] = processNTU_SKELETON_XView()

skel_work_dir_1 = '/media/jianl/disk3/Jian/Projects/NTU_End2End/Skeleton_CNN/CLASSIFICATION/SKEL_Descriptors_XView/SKEL_Output_fc6_4096_FC7_1024/';
skel_s_list_1 = dir([skel_work_dir_1 'ntu*']);

% skel_work_dir_2 = './SKEL_Output_COMPARE_Features_SuperPixel_5x5_36x36_Skeleton/';
skel_work_dir_2 = './SKEL_Output_COMPARE_Features_SuperPixel_5x5_36x36_Skel_Velo_Sandwich/';
skel_s_list_2 = dir([skel_work_dir_2 'ntu*']);

te = [];
tel = [];
tr = [];
trl = [];

% te = zeros( 20000, 114688);
% tr = zeros( 30000, 114688);

% te_start_idx = 1;
% tr_start_idx = 1;

for i = 1 : length(skel_s_list_2)
    fprintf('processing %s ...\n', skel_s_list_2(i).name);
    
    skel_trainData_mat = [skel_work_dir_2 skel_s_list_2(i).name '/trainData_view.mat'];
    skel_trainLab_mat = [skel_work_dir_2 skel_s_list_2(i).name '/trainLab_view.mat'];
    skel_testData_mat = [skel_work_dir_2 skel_s_list_2(i).name '/testData_view.mat'];
    skel_testLab_mat = [skel_work_dir_2 skel_s_list_2(i).name '/testLab_view.mat'];
    load(skel_trainData_mat);% skel_trainData_2 = trainData; clear trainData;
    load(skel_trainLab_mat);% skel_trainLab_2 = trainLab; clear trainLab;
    load(skel_testData_mat); skel_testData_2 = testData; clear testData;
    load(skel_testLab_mat); skel_testLab_2 = testLab; clear testLab;

    random_idx = reshape(randi(size(trainData,1), uint16(sqrt(size(trainData,1)*0.70))), [], 1);
    skel_trainData_2 = trainData(random_idx,:);
    skel_trainLab_2 = trainLab(random_idx,:);
    
        tr = [tr; skel_trainData_2];
%         tr(tr_start_idx : tr_start_idx+size(skel_trainData,1)-1 , :) = skel_trainData;
        trl = [trl; skel_trainLab_2(:,5)];
        te = [te; skel_testData_2];
%         te(te_start_idx : te_start_idx+size(skel_testData,1)-1 , :) = skel_testData;
        tel = [tel; skel_testLab_2(:,5)];
    
%     tr_start_idx = tr_start_idx + size(skel_trainData,1);
%     te_start_idx = te_start_idx + size(skel_testData,1);
    
end


% tr(tr_start_idx:end, :) = [];
% te(te_start_idx:end, :) = [];

n_classes =60;

trl_onehot = zeros( size( trl, 1 ), n_classes);
for i = 1:n_classes
    rows = trl == i;
    trl_onehot( rows, i ) = 1;
end

tel_onehot = zeros( size( tel, 1 ), n_classes);
for i = 1:n_classes
    rows = tel == i;
    tel_onehot( rows, i ) = 1;
end

DataBase.testing_feats = te';
DataBase.training_feats = normc(tr');
DataBase.H_train = trl_onehot';
DataBase.H_test = tel_onehot';
end

