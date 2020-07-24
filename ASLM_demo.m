
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A demo of ASLM regressio on Lorenz data set
%Zhengda Qin
%2019-5-20 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Dimension = 10;%embedding
N_tr = 10000;  %training set size
N_te = 400;    %testing set size
inputRow = 3;  %input axis of Lorenz data
desireRow = 3; %desire axis of Lorenz data
np = 0.0;      %noise
    
%======data formatting===========
load('lorenz.mat')  
lorenz = bsxfun(@minus,lorenz,mean(lorenz,2));
lorenz = bsxfun(@rdivide,lorenz,std(lorenz,[],2));

% training data
train_set = lorenz;
Input_train = zeros(Dimension,N_tr);
for k=1:N_tr
    Input_train(:,k) = train_set(inputRow,k:k+Dimension-1)';
end
Desire_train = train_set(desireRow,Dimension+1:Dimension+N_tr);
Desire_train = Desire_train+np*randn(size(Desire_train));

% testing data after the training data
test_set = lorenz(:,Dimension+N_tr+1:Dimension+N_tr+N_te+100);
Input_test = zeros(Dimension,N_te);
for k=1:N_te
    Input_test(:,k) = test_set(inputRow,k:k+Dimension-1)';
end
Desire_test = test_set(desireRow,Dimension+1:Dimension+N_te);
%======end of data formatting===========

%===============LS================
tic;
regular_LS = 0.1;    %regularization factor
Input_train_LS = [Input_train;ones(1,length(Input_train))];
Input_test_LS = [Input_test;ones(1,length(Input_test))];
LSweight = (eye(Dimension+1)*regular_LS+Input_train_LS*Input_train_LS')^-1*Input_train_LS*Desire_train';
trainingTime = toc;
tic;
mse_LS_tr = mean((Desire_train-LSweight'*Input_train_LS).^2);
mse_LS_te = mean((Desire_test-LSweight'*Input_test_LS).^2);
testingTime = toc;
fprintf('LS: training MSE = %.4f, testing MSE = %.4f, training Time = %.4f, testing Time = %.4f;\n',...
    mse_LS_tr, mse_LS_te, trainingTime, testingTime) 

%=============ASLM===================
Kpara = 1;   %parameter k in K-nearest neighbor
[mse_ASLM_tr,~,mse_ASLM_te,~,time_ASLM_tr,time_ASLM_te] = ...
    ASLM(Input_train,Desire_train,Input_test,Desire_test,LSweight,Kpara);
fprintf('ASLM: training MSE = %.4f, testing MSE = %.4f, training Time = %.4f, testing Time = %.4f; \n',...
    mse_ASLM_tr, mse_ASLM_te,time_ASLM_tr,time_ASLM_te) 

%=============KNN====================
VQradius = 0;   
[mse_KNN_tr,~,mse_KNN_te,~,~,time_KNN_tr,time_KNN_te] = ...
    KNN(Input_train,Desire_train,Input_test,Desire_test,VQradius,Kpara);
fprintf('KNN: training MSE = %.4f, testing MSE = %.4f, training Time = %.4f, testing Time = %.4f; \n',...
    mse_KNN_tr, mse_KNN_te,time_KNN_tr,time_KNN_te) 

% %=============KLMS====================
stepSize = 0.5;
VQradius = 0;
kernelSize = 1;
[mse_QKLMS_te,~,~,~,~,mse_QKLMS_tr,~,time_KLMS_tr,time_KLMS_te] = ...
    QKLMS(Input_train,Desire_train,Input_test,Desire_test,stepSize,VQradius,kernelSize);
fprintf('KLMS: training MSE = %.4f, testing MSE = %.4f, training Time = %.4f, testing Time = %.4f; \n',...
    mse_QKLMS_tr, mse_QKLMS_te,time_KLMS_tr,time_KLMS_te) 
    
%================KLMS-ASM================
[mse_ASM_te,time_ASM_tr,time_ASM_te] = KLMS_ASM(center,Input_test,Desire_test,y_KLMS_te,e_KLMS_tr);
fprintf('KLMS-ASM: training MSE = %.4f, testing MSE = %.4f, training Time = %.4f, testing Time = %.4f; \n',...
    0, mse_ASM_te,time_KLMS_tr+time_ASM_tr,time_KLMS_te+time_ASM_te) 
