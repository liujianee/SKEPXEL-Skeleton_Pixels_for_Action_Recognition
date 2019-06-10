clear all
clc

%SKEL_Descriptors = './Descriptors_FTP_Corr_Strong_to_Weak/';
%SKEL_Outputs = 'SKEL_Output_Corr_Strong_to_Weak/';

SKEL_Descriptors = './Descriptors_FTP_SuperPixel_5x5_36_Skeleton/';
SKEL_Outputs = 'SKEL_Output_SuperPixel_5x5_36_Skeleton/';

%%%protocol definition
train_subject_id = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38];
test_subject_id = [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40];
train_view_id = [2,3];
test_view_id = [1];

s_list = dir([SKEL_Descriptors 'ntu*']);

%%
for i = 8 : length(s_list)
    fprintf('processing %s ...\n', s_list(i).name);
    %s_dirs_batch = s_list(i).name;
    label_file = [SKEL_Descriptors s_list(i).name '/' 'Labels_114688.mat'];
    feature_file = [SKEL_Descriptors s_list(i).name '/' 'final_features_114688.mat'];
    load(label_file)
    load(feature_file)
    view = labels(:,2);
    subject = labels(:,3);
    % cross subject
    ind_sub_train = [];
    for sub_idx = 1 : length(train_subject_id)
        ind_sub_train = [ind_sub_train; find(subject == train_subject_id(sub_idx))];
    end
    
    ind_sub_test = [];
    for sub_idx = 1 : length(test_subject_id)
        ind_sub_test = [ind_sub_test; find(subject == test_subject_id(sub_idx))];
    end
 
    % cross view
    ind_view_train = [];
    for view_idx = 1 : length(train_view_id)
        ind_view_train = [ind_view_train; find(view == train_view_id(view_idx))];
    end    
    
    ind_view_test = [];
    for view_idx = 1 : length(test_view_id)
        ind_view_test = [ind_view_test; find(view == test_view_id(view_idx))];
    end     
    
    trainSamples_sub = final_features(ind_sub_train, :);
    trainLabels_sub = labels(ind_sub_train, :);
    testSamples_sub = final_features(ind_sub_test, :);
    testLabels_sub = labels(ind_sub_test, :);
    
    trainSamples_view = final_features(ind_view_train, :);
    trainLabels_view = labels(ind_view_train, :);   
    testSamples_view = final_features(ind_view_test, :);
    testLabels_view = labels(ind_view_test, :); 
    
    %trainSamples_sub = scaleDescs(trainSamples_sub);    
    %testSamples_sub = scaleDescs(testSamples_sub);
    %trainSamples_view = scaleDescs(trainSamples_view);    
    %testSamples_view = scaleDescs(testSamples_view);
    
    %% save SKEL output
    skel_output_dir = [SKEL_Outputs s_list(i).name];
    if ~exist(skel_output_dir,'dir')
        mkdir(skel_output_dir);
    end
    % cross subject
    trainData = trainSamples_sub;
    trainLab = trainLabels_sub;
    testData = testSamples_sub;
    testLab = testLabels_sub;
    save([skel_output_dir '/trainData_sub'], 'trainData' ,'-v7.3');
    save([skel_output_dir '/trainLab_sub'], 'trainLab');
    save([skel_output_dir '/testData_sub'], 'testData' ,'-v7.3');
    save([skel_output_dir '/testLab_sub'], 'testLab');
    
    % cross view
    trainData = trainSamples_view;
    trainLab = trainLabels_view;
    testData = testSamples_view;
    testLab = testLabels_view;
    save([skel_output_dir '/trainData_view'], 'trainData' ,'-v7.3');
    save([skel_output_dir '/trainLab_view'], 'trainLab');
    save([skel_output_dir '/testData_view'], 'testData' ,'-v7.3');
    save([skel_output_dir '/testLab_view'], 'testLab');   
       
    
    fprintf('SKEL_Output %s is done!\n', s_list(i).name);          

end

