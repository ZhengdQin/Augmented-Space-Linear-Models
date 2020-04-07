function [WCenter,WCenNum,centerInclude,dataLabel,quantizeError] = VQ(W,epsilon,error)
Dimension = size(W,1);
D = size(W,2);
WCenter = zeros(Dimension,D);

%initial
WCenter(:,1) = W(:,1);
WCenNum = 1;
centerInclude = zeros(D,1);
centerInclude(1) = 1;
quantizeError = zeros(D,1);
quantizeError(1) = error(1);
dataLabel= ones(D,1);
for i = 2:D
    dimension = sum((W(:,i)*ones(1,WCenNum)-WCenter(:,1:WCenNum)).^2);
    [va,index] = min(dimension);
%     [index,va] = knnsearch(WCenter(:,1:WCenNum)',W(:,i)');
    if va >= epsilon
        WCenNum = WCenNum+1;
        WCenter(:,WCenNum) = W(:,i);
        centerInclude(WCenNum) = 1;
        quantizeError(WCenNum) = quantizeError(WCenNum)+error(i);
        dataLabel(i) = WCenNum;
    else
        centerInclude(index) = centerInclude(index)+1;
        quantizeError(index) = quantizeError(index)+error(i);
        dataLabel(i) = index;
    end
end
WCenter = WCenter(:,1:WCenNum);
end