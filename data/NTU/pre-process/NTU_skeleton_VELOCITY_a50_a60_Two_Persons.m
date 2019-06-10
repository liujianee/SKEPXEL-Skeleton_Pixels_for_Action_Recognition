clear
close all

skelWork = './nturgb+d_skeletons_extract_clean/';
saveDir = './nturgb+d_skeletons_VELOCITY_a50_a60_Two_Persons/';

ac_Files = dir([skelWork 'S*']);

if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end   

%%
for ac = 1:length(ac_Files)
    fprintf('processing %s ...\n', ac_Files(ac).name);
    C = strsplit(ac_Files(ac).name, 'A');
    CC = strsplit(C{2}, '.');
    acion_ID = str2num(CC{1});
    skelFile = [skelWork ac_Files(ac).name];

%     load(skelFile) 
    if exist(skelFile, 'file')
        load(skelFile)  
    else
        fprintf('...bad skeleton file, continue ...\n')
        continue;
    end
    fm_num = size(sk,4);   
    
    if (acion_ID > 49)
              
        skel_A = reshape(sk(:,1:3,1,:), [25, 3, fm_num]);
        skel_B = reshape(sk(:,1:3,2,:), [25, 3, fm_num]);
        
        skel_A_velocity = skel_A(:,:,2:end) - skel_A(:,:,1:end-1);
        skel_B_velocity = skel_B(:,:,2:end) - skel_B(:,:,1:end-1);
        
        fm_num_v = size(skel_A_velocity, 3);
        
        sk_out = zeros(25, 3, fm_num_v * 2);
        ind_A = 1: 2: fm_num_v*2;
        sk_out(:,:,ind_A) = skel_A_velocity;
        ind_B = 2: 2: fm_num_v*2;
        sk_out(:,:,ind_B) = skel_B_velocity;
        
        del_index = [];
        for index = 1 : fm_num_v*2
            if sum(sum(sk_out(:,:,index))) == 0
                fprintf('delete empty skeleton for %s \n', ac_Files(ac).name)
                del_index = [del_index; index];
                % remove more related index
                if index - 2 > 0
                    del_index = [del_index; index-2];
                end
                if index + 2 < fm_num_v*2 +1
                    del_index = [del_index; index+2];
                end
            end
        end
        del_index = unique(del_index);
        sk_out(:,:,del_index) = [];
        sk_out_velocity = sk_out;
        
    else
        sk_out = reshape(sk(:,1:3,1,:), [25, 3, fm_num]);
        sk_out_velocity = sk_out(:,:,2:end) - sk_out(:,:,1:end-1);        
    end

    sk = sk_out_velocity;
    saveFile = [saveDir ac_Files(ac).name];
    save(saveFile, 'sk');   
end
