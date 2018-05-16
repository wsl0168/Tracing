% This function load a trained network and test it on a 3D image stack
% Using FFT to speed up convn function

% Load network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[FILENAME, PATHNAME]=uigetfile();
load(strcat(PATHNAME,FILENAME));

% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stack_test = ind_test_image;
data_path=['..\data\',num2str(stack_test),'_L6_AS.mat'];
data_dist_path=['..\data\',num2str(stack_test),'_L6_AS_Max9_round_3D.mat'];

% Test Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data=load(data_path);
ImIm = data.IM;
ImIm = double(ImIm);
ImIm = ImIm./255;

DD = load(data_dist_path);
DD = DD.D;

Jm = ImIm;
ImIm = zeros(size(Jm) + [2*bound_xy 2*bound_xy 2*bound_z]);
ImIm(bound_xy+1:end-bound_xy,bound_xy+1:end-bound_xy,bound_z+1:end-bound_z) = Jm;
JD = DD;
DD = inf(size(JD) + [2*bound_xy 2*bound_xy 2*bound_z]);
DD(bound_xy+1:end-bound_xy,bound_xy+1:end-bound_xy,bound_z+1:end-bound_z) = JD;
clear Jm JD

a = (winsize_xy+1)/2;
b = size(DD,1)-(winsize_xy-1)/2;
a_z = (winsize_z+1)/2;
b_z = size(DD,3)-(winsize_z-1)/2;

% Test Set
Im_test = ImIm(a:b,a:b,a_z:b_z);

% Calculate test result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=cell(1,N_layers);
filter=zeros(winsize_xy,winsize_xy,winsize_z);
[X,Y,Z] = size(ImIm);
Im_temp = fftn(ImIm);
for i=1:size(W{1},1)
    filter = reshape(W{1}(i,:),[winsize_xy,winsize_xy,winsize_z]);
    filter(X,Y,Z) = 0;
    temp_train = ifftn(Im_temp.*fftn(filter));
    temp_train = temp_train(winsize_xy:X,winsize_xy:Y,winsize_z:Z)+bias*B{1}(i,1);
    V{1}(i,:)=1./(1+exp(-(beta).*temp_train(:)));
end

for i = 2:N_layers
    V{i} = 1./(1+exp(-(beta).*(W{i}*V{i-1}+bias*repmat(B{i},1,size(V{i-1},2)))));
end

Out=zeros(size(Im_test));
Out(:)=V{end};

label_test=DD(a:b,a:b,a_z:b_z);

figure,imshow(max(Im_test,[],3))
figure,imshow(max(Out,[],3))
figure,imshow(max(label_test<=1,[],3))

clear V
save([PATHNAME,'Q',FILENAME,'.mat']);