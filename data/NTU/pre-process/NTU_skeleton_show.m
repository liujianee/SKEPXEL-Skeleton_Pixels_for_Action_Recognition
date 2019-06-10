
skel_dir = '/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_norm/';
% skel_dir = '/media/jianl/disk3/Jian/Datasets/NTU/nturgb+d_skeletons_matfile/';

skel_Files = dir([skel_dir 'S001C001P001R001*']);


% this is for NTU skeleton data
skpoly = [1 2 1; 1 13 1; 13 14 13; 14 15 14; 15 16 15; 1 17 1; 17 18 17; ...
    18 19 18; 19 20 19; 2 21 2; 21 5 21; 5 6 5; 6 7 6; 7 8 7; 8 23 8; ...
    8 22 8; 21 9 21; 9 10 9; 10 11 10; 11 12 11; 12 25 12; 12 24 12; 21 3 21; 3 4 3];

%%

for s_idx = 6:6%1:size(skel_Files, 1)
    skel_file = [skel_dir skel_Files(s_idx).name];
    load(skel_file);
    bone_length = zeros(size(sk,3), size(skpoly,1));
%     for f_idx = 1:size(sk,3)
%         for b_idx = 1:size(skpoly,1)
%             b_1 = skpoly(b_idx,1);
%             b_2 = skpoly(b_idx,2);
%             bone_length(f_idx, b_idx) = norm(sk(b_1,:,f_idx) - sk(b_2,:,f_idx));
%         end    
%     end
    for f_idx = 1:size(sk,3)
        clf
        pts = sk(:,:,f_idx);
        subplot(1,2,1), % display skeleton
        plot3(pts(:,1), pts(:,2),pts(:,3),'or'); axis on; grid on; axis equal; hold on;
        trimesh(skpoly,pts(:,1), pts(:,2),pts(:,3)); view(0,90);
        scatter3(pts(2,1), pts(2,2),pts(2,3),'filled');
        scatter3(pts(1,1), pts(1,2),pts(1,3),'filled');
        xlabel('x'); ylabel('y'); zlabel('z'); 
        subplot(1,2,2), % plot bone length changing
        for b_idx = 1: size(skpoly,1)
            b_1 = skpoly(b_idx,1);
            b_2 = skpoly(b_idx,2);
            bone_length(f_idx, b_idx) = norm(sk(b_1,:,f_idx) - sk(b_2,:,f_idx));
            plot(bone_length(:,b_idx)); hold on;
        end
        pause(0.1)
    end    
end

%%

for s_idx = 1: size(skel_Files, 1)
    skel_file = [skel_dir skel_Files(s_idx).name];
    load(skel_file);
    for f_idx = 1:size(sk,4)
        clf
        pts = sk(:,1:3,1,f_idx);
        %subplot(1,2,1), % display skeleton
        plot3(pts(:,1), pts(:,2),pts(:,3),'or'); axis on; grid on; axis equal; hold on;
        trimesh(skpoly,pts(:,1), pts(:,2),pts(:,3)); view(0,90);
        scatter3(pts(2,1), pts(2,2),pts(2,3),'filled');
        scatter3(pts(1,1), pts(1,2),pts(1,3),'filled');
        pts = sk(:,1:3,2,f_idx);
        %subplot(1,2,2), % display skeleton
        plot3(pts(:,1), pts(:,2),pts(:,3),'or'); axis on; grid on; axis equal; hold on;
        trimesh(skpoly,pts(:,1), pts(:,2),pts(:,3)); view(0,90);       
        
        pause(0.01)
    end    
end



    for f_idx = 1:size(sk,3)
        clf
        pts = sk(:,:,f_idx);
        %subplot(1,2,1), % display skeleton
        plot3(pts(:,1), pts(:,2),pts(:,3),'or'); axis on; grid on; axis equal; hold on;
        trimesh(skpoly,pts(:,1), pts(:,2),pts(:,3)); view(0,90);
        scatter3(pts(2,1), pts(2,2),pts(2,3),'filled');
        scatter3(pts(1,1), pts(1,2),pts(1,3),'filled');     
        pause(0.1)
    end  

