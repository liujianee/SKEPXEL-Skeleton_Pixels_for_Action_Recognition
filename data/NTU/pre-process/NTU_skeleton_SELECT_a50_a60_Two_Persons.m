clear
close all

skelWork = './nturgb+d_skeletons_extract_clean/';
saveDir = './nturgb+d_skeletons_SELECT_a50_a60_Two_Persons/';


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
        
        sk_out = zeros(25, 3, fm_num*2);
        ind_A = 1: 2: fm_num*2;
        sk_out(:,:,ind_A) = skel_A;
        ind_B = 2: 2: fm_num*2;
        sk_out(:,:,ind_B) = skel_B;
        
        del_index = [];
        for index = 1 : fm_num*2
            if sum(sum(sk_out(:,:,index))) == 0
                fprintf('delete empty skeleton\n')
                del_index = [del_index; index];
            end
        end
        sk_out(:,:,del_index) = [];
        
    else
        sk_out = reshape(sk(:,1:3,1,:), [25, 3, fm_num]);  
    end

    sk = sk_out;
    saveFile = [saveDir ac_Files(ac).name];
    save(saveFile, 'sk');   
end
