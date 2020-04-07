function [mse_JSLM_tr,output_tr,mse_JSLM_te,output_te,trainTime,testTime] = ASLM(Input_train,Desire_train,...
    Input_test,Desire_test,LSweight,para)

trainNum = size(Input_train,2);
testNum = size(Input_test,2);
Desire_tr = Desire_train';                   
outWeight = [-LSweight(1:end-1);1;-LSweight(end)]; 

tic;
%compute the error
input_tr = [Input_train',Desire_tr];
input_LS = [Input_train',zeros(size(Input_train',1),1),ones(trainNum,1)];
weight_LS = outWeight;
desireEstimate_LS = bsxfun(@times,input_LS,weight_LS');
desireEstimate = bsxfun(@times,[input_tr,ones(trainNum,1)],weight_LS');
errorVal = sum(desireEstimate,2);
Mdl = KDTreeSearcher(desireEstimate_LS);  %MATLAB
trainTime = toc;
% KDtree

%testing MSE
tic;
desireEstimate_LSte = bsxfun(@times,[Input_test',zeros(testNum,1),ones(testNum,1)],weight_LS');
clostSplNum = knnsearch(Mdl,desireEstimate_LSte,'k',para);
output_te = sum(errorVal(clostSplNum),2)-sum(desireEstimate_LSte,2);
mse_JSLM_te = mean((Desire_test-output_te').^2);
testTime = toc;
%training MSE
output_tr = errorVal(1:trainNum)-sum(desireEstimate_LS,2);
mse_JSLM_tr = mean((Desire_train-output_tr').^2);
end
