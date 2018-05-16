%{ 
This function train a completely connected neural network to solve image
processing problem.
Input of the network is a cubic in the 3D image stack, the output is a 
number between 0 and 1 that represents the probability
that a neuritepassing through the central voxel of the image.
Using Sigmoid function as activation functions.
Using mean square error as loss function.
The triained network will be saved in the save_path for being tested with 
ANN_test
%}


% function ABC_train_V1
format short g

% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save_path = '..\trained_networks\'; 
ind_validation_image = 4;
ind_test_image = 5;
ratio01 = 0; % if ratio01 == 0, not selecting training examples
thick_validation = 7;

% Creating Labeled Dataset
Max_Dist=3;
Min_Dist=1;

% Network architecture
winsize_xy = 21; % must be odd
winsize_z = 7; % must be odd
Layer_sizes=[36,1]; % must be all odd %% specify neuron numbers by how many neurons it connects to the previous layer?
bound_xy = (winsize_xy-1)/2;
bound_z = (winsize_z-1)/2;

% Training parameters
learningrate_ratio = [1 1 1 1 1 1];
learningrate_amp = 1;
learningrate = learningrate_ratio*learningrate_amp;
ini_weight_amp = [0.1 0.1 0.1 0.1 0.1 0.1];
beta = 1;
N_steps= 5*10^5;
region = 10^5;
output_period = 2*10^5;
bias = 1;
u = 255;

l_s_text = num2str(Layer_sizes);
l_s_text = l_s_text(~isspace(l_s_text));
lr_text = num2str(learningrate_amp);
lr_text = lr_text(~isspace(lr_text));
l_w_a_text = num2str(ini_weight_amp(1:length(Layer_sizes)));
l_w_a_text = l_w_a_text(~isspace(l_w_a_text));


% Training Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Im = cell(6,8);
D = cell(6,8);
for i = 1:6
    train_path=['..\data\',num2str(i),'_L6_AS.mat'];
    data=load(train_path);
    Im{i,1} = data.IM;
    Im{i,1} = double(Im{i,1});
    Im{i,1} = Im{i,1}./255;
    
    
    train_dist_path=['..\data\',num2str(i),'_L6_AS_Max9_round_3D.mat'];
    D{i,1} = load(train_dist_path);
    D{i,1} = D{i,1}.D;
    
    Jm = Im{i,1};
    Jm = Jm(round(size(Jm,1)/2)-u:round(size(Jm,1)/2)+u+1,round(size(Jm,2)/2)-u:round(size(Jm,2)/2)+u+1,:);
    Im{i,1} = zeros(size(Jm) + [2*bound_xy 2*bound_xy 2*bound_z]);
    Im{i,1}(bound_xy+1:end-bound_xy,bound_xy+1:end-bound_xy,bound_z+1:end-bound_z) = Jm;
    JD = D{i,1};
    JD = JD(round(size(JD,1)/2)-u:round(size(JD,1)/2)+u+1,round(size(JD,2)/2)-u:round(size(JD,2)/2)+u+1,:);
    D{i,1} = inf(size(JD) + [2*bound_xy 2*bound_xy 2*bound_z]);
    D{i,1}(bound_xy+1:end-bound_xy,bound_xy+1:end-bound_xy,bound_z+1:end-bound_z) = JD;
    clear Jm JD
    
end

% rotate and flip matrixs
for i = 1:6
    Im{i,2} = imrotate(Im{i,1},90);
    Im{i,3} = imrotate(Im{i,1},180);
    Im{i,4} = imrotate(Im{i,1},270);
    Im{i,5} = flip(Im{i,1},1);
    Im{i,6} = flip(Im{i,1},2);
    Im{i,7} = flip(imrotate(Im{i,1},90),1);
    Im{i,8} = flip(imrotate(Im{i,1},90),2);
    
    D{i,2} = imrotate(D{i,1},90);
    D{i,3} = imrotate(D{i,1},180);
    D{i,4} = imrotate(D{i,1},270);
    D{i,5} = flip(D{i,1},1);
    D{i,6} = flip(D{i,1},2);
    D{i,7} = flip(imrotate(D{i,1},90),1);
    D{i,8} = flip(imrotate(D{i,1},90),2);
end

% producing training input and output.
% We use 3 types of voxels labeled 1 (axon), 0 (near the axon), and 0 (in the background)
% Choosing axon voxels
for i = 1:6
    ind_1=find(D{i,1}(:)<=Min_Dist);
    [ii,jj,kk]=ind2sub(size(D{i,1}),ind_1);
    ind_1=ind_1(ii>=(winsize_xy+1)/2 & ii<=size(D{i,1},1)-(winsize_xy+1)/2+1 & ...
        jj>=(winsize_xy+1)/2 & jj<=size(D{i,1},2)-(winsize_xy+1)/2+1 & ...
        kk>=(winsize_z+1)/2 & kk<=size(D{i,1},3)-(winsize_z+1)/2+1);
    ind_1=ind_1(randperm(length(ind_1)));
    [ii_1,jj_1,kk_1]=ind2sub(size(D{i,1}),ind_1);
    
    ind_2=find(D{i,1}(:)>Max_Dist);
    [ii,jj,kk]=ind2sub(size(D{i,1}),ind_2);
    ind_2=ind_2(ii>=(winsize_xy+1)/2 & ii<=size(D{i,1},1)-(winsize_xy+1)/2+1 & ...
        jj>=(winsize_xy+1)/2 & jj<=size(D{i,1},2)-(winsize_xy+1)/2+1 & ...
        kk>=(winsize_z+1)/2 & kk<=size(D{i,1},3)-(winsize_z+1)/2+1);
    ind_2=ind_2(randperm(length(ind_2)));
    
    if ratio01 ~= 0
        ind_2=ind_2(1:ratio01*length(ind_1)); % set ratio of 1 and 0
    end
    
    [ii_2,jj_2,kk_2]=ind2sub(size(D{i,1}),ind_2);
    
    ind_12{i}=[ind_1;ind_2];
    ii_12{i}=[ii_1;ii_2];
    jj_12{i}=[jj_1;jj_2];
    kk_12{i}=[kk_1;kk_2];
end

% rotation and flip matrix
ii_rot = cell(6,8);
jj_rot = cell(6,8);
kk_rot = cell(6,8);
for i=1:6
    kai = reshape(1:numel(D{i,1}),size(D{i,1}));
    [ii_rot{i,1},jj_rot{i,1},kk_rot{i,1}]= ind2sub(size(D{i,1}),kai);
    [ii_rot{i,2},jj_rot{i,2},kk_rot{i,2}] = ind2sub(size(D{i,1}),imrotate(kai,-90));
    [ii_rot{i,3},jj_rot{i,3},kk_rot{i,3}] = ind2sub(size(D{i,1}),imrotate(kai,-180));
    [ii_rot{i,4},jj_rot{i,4},kk_rot{i,4}] = ind2sub(size(D{i,1}),imrotate(kai,-270));
    [ii_rot{i,5},jj_rot{i,5},kk_rot{i,5}] = ind2sub(size(D{i,1}),flip(kai,1));
    [ii_rot{i,6},jj_rot{i,6},kk_rot{i,6}] = ind2sub(size(D{i,1}),flip(kai,2));
    [ii_rot{i,7},jj_rot{i,7},kk_rot{i,7}] = ind2sub(size(D{i,1}),flip(imrotate(kai,90),1));
    [ii_rot{i,8},jj_rot{i,8},kk_rot{i,8}] = ind2sub(size(D{i,1}),flip(imrotate(kai,90),2));
end

% Create Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convolutional Network Structure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_neurons=Layer_sizes;
N_layers=length(N_neurons); % not including Im

% Initialize Weights %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = cell(1,N_layers);
B = cell(1,N_layers);
W_save = cell(1,N_steps/region+1);
B_save = cell(1,N_steps/region+1);

W{1} = 2*ini_weight_amp(1)*(rand(N_neurons(1),winsize_xy*winsize_xy*winsize_z)-0.5);
B{1} = 2*ini_weight_amp(1)*(rand(size(W{1},1),1)-0.5);
for i = 2 : N_layers
    W{i} = 2*ini_weight_amp(i)*(rand(N_neurons(i),N_neurons(i-1))-0.5);
    B{i} = 2*ini_weight_amp(i)*(rand(size(W{i},1),1)-0.5);
end

% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iteration=0;
loss=zeros(1,N_steps/region+1);

V=cell(1,N_layers);

ind_Zeta = cell(1,8);
filter=zeros(winsize_xy,winsize_xy,winsize_z);
for i=1:size(W{1},1)
    filter(:)=W{1}(i,:);
    temp_validation=convn(Im{ind_validation_image,1}(:,:,1:thick_validation),filter,'valid')+bias*B{1}(i,1);
    V{1}(i,:)=1./(1+exp(-(beta).*temp_validation(:)));
end
temp_Zeta = D{ind_validation_image,1}((winsize_xy+1)/2:size(D{ind_validation_image,1},1)-(winsize_xy-1)/2,(winsize_xy+1)/2:size(D{ind_validation_image,1},2)-(winsize_xy-1)/2,(winsize_z+1)/2:thick_validation-(winsize_z-1)/2);%size(D{j},3)-(winsize_z-1)/2);
Zeta_big = temp_Zeta(:)';
ind_Zeta{1} = find(Zeta_big <= Min_Dist | Zeta_big > Max_Dist);
V{1} = V{1}(:,ind_Zeta{1});
Zeta_big = Zeta_big(ind_Zeta{1});
Zeta_big = (Zeta_big <= Min_Dist);
Zeta_temp=zeros(1,8*size(Zeta_big,2));
Zeta_temp(1,1:size(Zeta_big,2)) = Zeta_big;
for j=2:8
    filter=zeros(winsize_xy,winsize_xy,winsize_z);
    filter_info=zeros(size(W{1},1),numel(temp_validation));
    for i=1:size(W{1},1)
        filter(:)=W{1}(i,:);
        temp_validation=convn(Im{ind_validation_image,j}(:,:,1:thick_validation),filter,'valid')+bias*B{1}(i,1);
        filter_info(i,:)=1./(1+exp(-(beta).*temp_validation(:)));
    end
    temp_Zeta = D{ind_validation_image,j}((winsize_xy+1)/2:size(D{ind_validation_image,j},1)-(winsize_xy-1)/2,(winsize_xy+1)/2:size(D{ind_validation_image,j},2)-(winsize_xy-1)/2,(winsize_z+1)/2:thick_validation-(winsize_z-1)/2);%size(D_copy{j},3)-(winsize_z-1)/2);
    Zeta_big = temp_Zeta(:)';
    ind_Zeta{j} = find(Zeta_big <= Min_Dist | Zeta_big > Max_Dist);
    filter_info = filter_info(:,ind_Zeta{j});
    Zeta_big = Zeta_big(ind_Zeta{j});
    Zeta_big = (Zeta_big <= Min_Dist);
    V{1} = [V{1},filter_info];
    Zeta_temp(1,(j-1)*size(Zeta_big,2)+1:j*size(Zeta_big,2)) = Zeta_big;
end
for i = 2:N_layers
    V{i} = 1./(1+exp(-(beta).*(W{i}*V{i-1}+bias*B{i}*ones(1,size(V{i-1},2)))));
end
Zeta_big = Zeta_temp;

loss(1) = mean((V{end}-Zeta_big).^2);
disp([iteration,loss(1),learningrate_amp])

V_mu0=cell(1,N_layers);
W_max=W;
Delta=cell(1,N_layers);
Delta_W = cell(1,N_layers); % for calculating gradient and update W;

W_save{1} = W;
B_save{1} = B;

tic
while iteration<=N_steps
    iteration=iteration+1;
    
    
    image_ind=randi(6);
    while (image_ind == ind_test_image) || (image_ind == ind_validation_image)
        image_ind=randi(6);
    end
    mu0_ind=randi(length(ind_12{image_ind}));
    rot_ind=randi(8);
    
    mu0=ind_12{image_ind}(mu0_ind);
    Xi=Im{image_ind,rot_ind}(ii_rot{image_ind,rot_ind}(mu0)-(winsize_xy-1)/2:ii_rot{image_ind,rot_ind}(mu0)+(winsize_xy-1)/2,jj_rot{image_ind,rot_ind}(mu0)-(winsize_xy-1)/2:jj_rot{image_ind,rot_ind}(mu0)+(winsize_xy-1)/2,kk_rot{image_ind,rot_ind}(mu0)-(winsize_z-1)/2:kk_rot{image_ind,rot_ind}(mu0)+(winsize_z-1)/2);
    Zeta_mu0 = (D{image_ind,rot_ind}(ii_rot{image_ind,rot_ind}(mu0),jj_rot{image_ind,rot_ind}(mu0),kk_rot{image_ind,rot_ind}(mu0))<=Min_Dist);
    
    % feed forward
    V_mu0{1} = 1./(1+exp(-(beta).*(W{1}*Xi(:)+bias*B{1})));
    for i = 2:N_layers-1
        V_mu0{i} = 1./(1+exp(-(beta).*(W{i}*V_mu0{i-1}+bias*B{i})));
    end
    V_mu0{N_layers} = 1./(1+exp(-(beta).*(W{N_layers}*V_mu0{N_layers-1}+bias*B{N_layers})));
    
    % backprop
    Delta{N_layers} =beta*V_mu0{N_layers}.*(1-V_mu0{N_layers}).*(Zeta_mu0-V_mu0{N_layers})';
    Delta_W{N_layers} = Delta{N_layers}*V_mu0{N_layers-1}';
    for i = N_layers-1:-1:2
        Delta{i} = beta*V_mu0{i}.*(1-V_mu0{i}).*(W{i+1}'*Delta{i+1});
        Delta_W{i} = Delta{i}*V_mu0{i-1}';
    end
    Delta{1} = beta*V_mu0{1}.*(1-V_mu0{1}).*(W{2}'*Delta{2});
    Delta_W{1} = Delta{1}*Xi(:)';
    
    B{1} = B{1} + learningrate(1)*Delta{1}*bias;
    grad = Delta_W{1};
    W{1} = W{1} + learningrate(1)*grad;
    
    for i = 2:N_layers
        W{i} = W{i} + learningrate(i)*Delta_W{i};
        B{i} = B{i} + learningrate(i)*Delta{i}*bias;
    end
    
    % output current training result %%%%%%%%%%%%%%%%%%%
    if mod(iteration,region)==0
        
        if mod(iteration,output_period)==0
            elapse_time = toc;
            variables = who;
            toexclude = {'V','filter_info','Im','D','Im_copy','D_copy','ind_12','ii_12','jj_12','kk_12','ii','jj','kk','ii_1','jj_1','kk_1','ind_1','ii_2','jj_2','kk_2','ind_2','ti','Max_Xi'};
            variables = variables(~ismember(variables, toexclude));
            save(strcat(save_path,'ANN','_learningrate',lr_text,'_',num2str(iteration),'steps','_toc',num2str(elapse_time),'_8','.mat'),variables{:});
        end
        
        W_save{iteration/region+1} = W;
        B_save{iteration/region+1} = B;
        
        for j=1:8
            filter=zeros(winsize_xy,winsize_xy,winsize_z);
            filter_info=zeros(size(W{1},1),numel(temp_validation));
            for i=1:size(W{1},1)
                filter(:)=W{1}(i,:);
                temp_validation=convn(Im{ind_validation_image,j}(:,:,1:thick_validation),filter,'valid')+bias*B{1}(i,1);
                filter_info(i,:)=1./(1+exp(-(beta).*temp_validation(:)));
            end
            V{1}(:,(j-1)*size(Zeta_big,2)/8+1:(j)*size(Zeta_big,2)/8) = filter_info(:,ind_Zeta{j});
        end
        for i = 2:N_layers
            V{i} = 1./(1+exp(-(beta).*(W{i}*V{i-1}+bias*B{i}*ones(1,size(V{i-1},2)))));
        end
        loss(iteration/region+1) = mean((V{end}-Zeta_big).^2);
        disp([iteration,loss(iteration/region+1),learningrate_amp])
        
    end
end
toc

elapse_time = toc;
clear V filter_info Im D Im_copy D_copy ind_12 ii_12 jj_12 kk_12 ii jj kk ii_1 jj_1 kk_1 ind_1 ii_2 jj_2 kk_2 ind_2 ti Max_Xi kai data ind_Zeta Zeta_big Zeta_temp
save(strcat(save_path,'ANN','_learningrate',lr_text,'_',num2str(N_steps),'steps','_toc',num2str(elapse_time),'_8','.mat'));
