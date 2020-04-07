
function [mse_te_QKLMS,Center,Center_Num,quantzNum,y_te_Q,mse_tr_QKLMS,y_tr_Q,trainTime,testTime] = QKLMS(Input_train,Desire_train,Input_test,Desire_test,lr_k_Q,radius_1,initialdelta)

Dimension = size(Input_train,1); 
N_tr = size(Input_train,2);
N_te = size(Input_test,2);

%init QKLMS
e_k_Q = zeros(N_tr,1);
e_tr_Q = zeros(N_tr,1);
y_Q = zeros(N_tr,1);
Center = zeros(Dimension,N_tr);
kernelsize = 2*initialdelta^2;
e_quantz = zeros(N_tr,N_tr,2);
e_vary = zeros(N_tr,N_tr);
quantzNum = zeros(N_tr,1);
trainTime = 0;
% n=1 init
e_k_Q(1) = Desire_train(1);
y_Q(1) = 0;
Center(:,1) = Input_train(:,1);
Center_Num = 1;
e_quantz(Center_Num,1,1) = e_k_Q(1);
e_vary(Center_Num,1) = e_k_Q(1);
quantzNum(1) = 1; 

% start
for n=2:N_tr
    
tic;
    %training
    Kernel = sum((Input_train(:,n)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2);
    y_Q(n) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-Kernel./kernelsize))';
    e_tr_Q(n) = Desire_train(n)-y_Q(n);
    [va,index] = min(Kernel);
    if va >= radius_1
         Center_Num = Center_Num+1;
         Center(:,Center_Num) = Input_train(:,n);
         e_k_Q(Center_Num) = e_tr_Q (n);
         e_quantz(Center_Num,1,1) = e_tr_Q(n);
         e_vary(Center_Num,1) = e_tr_Q(n);
         quantzNum(Center_Num) = 1;
     else
         e_k_Q(index) = e_k_Q(index)+e_tr_Q (n);
         quantzNum(index) = quantzNum(index)+1;
         e_quantz(index,quantzNum(index),1) = e_tr_Q(n);
         e_vary(index,quantzNum(index)) = e_k_Q(index);
         e_quantz(index,quantzNum(index),2) = va;
     end
trainTime = toc+trainTime; 
    
end
%testing MSE
tic
y_te_Q = zeros(N_te,1);
for j = 1:N_te
    y_te_Q(j) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-sum((Input_test(:,j)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2)./kernelsize))';
end
err_Q = Desire_test -y_te_Q';
mse_te_QKLMS = mean(err_Q.^2);
testTime = toc;
y_tr_Q = zeros(N_tr,1);
for jj = 1:N_tr
    y_tr_Q(jj) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-sum((Input_train(:,jj)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2)./kernelsize))';
end
err_KLMS_tr = Desire_train -y_tr_Q';
mse_tr_QKLMS = mean(err_KLMS_tr.^2);
% e_changeAll = e_quantz(1:Center_Num,1:max(quantzNum),:);
% e_varyAll = e_vary(1:Center_Num,1:max(quantzNum));
end