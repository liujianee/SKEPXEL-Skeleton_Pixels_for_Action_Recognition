function skel_new = NTU_skeleton_normalize(skel)
% first translate to base_spine, and align y-axis to the spine direction;
% then rotate along y-axis to align x-axis to the direction of shoulder;
%%
% origin is in base of spine
base_spine = skel(1,:);
skel = skel - repmat(base_spine, [size(skel,1),1]);

%%
base_spine = skel(1,:); % new base_spine location
middle_spine = skel(2,:);

vec_spine = middle_spine - base_spine;

%%
local_xyz = vec_spine;
x = local_xyz(1);
y = local_xyz(2);
z = local_xyz(3);
[azimuth,elevation,r] = cart2sph(x,y,z);

azimuth = radtodeg(azimuth);
elevation = radtodeg(elevation);

% firstly rotate along z-axis
rxyz = [0; 0; 90-azimuth];
order = [3;2;1];
displ = [0;0;0];
transM_1 = transformation_matrix(displ,rxyz,order);
% secondly rotate along x-axis
rxyz = [-elevation; 0; 0];
order = [3;2;1];
displ = [0;0;0];
transM_2 = transformation_matrix(displ,rxyz,order);

transM = transM_2 * transM_1;

skel_1 = [skel, ones(size(skel,1),1)]';
skel_2 = transM * skel_1;
skel_2 = skel_2';

skel_3 = skel_2(:,1:3);

% pts = skel_3;
% plot3(pts(:,1), pts(:,2),pts(:,3),'or'); axis on; grid on; axis equal; hold on;
% trimesh(skpoly,pts(:,1), pts(:,2),pts(:,3)); xlabel('X'); ylabel('Y'); zlabel('Z'); view(0,90);


%%

right_shoulder = skel_3(9,:);
left_shoulder = skel_3(5,:);

vec_shoulder = left_shoulder - right_shoulder;
% u = [vec_shoulder(1), 0, vec_shoulder(3)]; % project vec_1 to x-z axis plane
% v = [1, 0, 0];  % unit vector along x-axis
% CosTheta = dot(u,v)/(norm(u)*norm(v));
% ThetaInDegrees = acosd(CosTheta);

x1 = vec_shoulder(1); y1 = vec_shoulder(3);
x2 = 1; y2 = 0;
ThetaInDegrees = atan2d(x1*y2-y1*x2,x1*x2+y1*y2);

rxyz = [0, -ThetaInDegrees, 0];% rotate the degree along y-axis
order = [3 2 1];
displ = [0 0 0]';
transM = transformation_matrix(displ,rxyz,order);

skel_4 = [skel_3, ones(size(skel_3,1),1)]';
skel_5 = transM * skel_4;
skel_5 = skel_5';

skel_6 = skel_5(:,1:3);

%%

skel_new = skel_6;


end
