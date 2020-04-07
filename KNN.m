function [mse_JSLM_tr,output_tr,mse_JSLM_te,output_te,centerNum,trainTime,testTime] = KNN(Input_train,Desire_train,...
    Input_test,Desire_test,VQradius,para)

Desire_tr = Desire_train';                  



tic;
errorVal = Desire_tr;
if VQradius==0
    center = Input_train;
    centerNum = length(Input_train);
    dataLabel = 1:centerNum;
    centerErrVal = errorVal;
else
[center,centerNum,centerInclude,dataLabel,centerErrVal] = ...
    VQ(Input_train,VQradius,errorVal);
centerErrVal = centerErrVal./centerInclude;
end
Mdl = KDTreeSearcher(center');  
trainTime = toc;


%testing MSE
tic;
clostSplNum = knnsearch(Mdl,Input_test','k',para); 
output_te = centerErrVal(clostSplNum);
mse_JSLM_te = mean((Desire_test-output_te').^2);
testTime = toc;
%training MSE
output_tr = centerErrVal(dataLabel);
mse_JSLM_tr = mean((Desire_train-output_tr').^2);
end

