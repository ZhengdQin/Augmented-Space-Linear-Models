function [MSE_ASM_te,trTime,teTime]=KLMS_ASM(center,Input_test,Desire_test,y_KLMS,err_KLMS_tr)
    tic;
    Mdl = KDTreeSearcher(center');
    trTime = toc;
%     KNNpara = Dimension;
    tic;
    clostSplNum = knnsearch(Mdl,Input_test','k',7);
    output_teKLMSAM = mean(err_KLMS_tr(clostSplNum),2)+y_KLMS;
    MSE_ASM_te = mean((Desire_test-output_teKLMSAM').^2);
    teTime = toc;
end