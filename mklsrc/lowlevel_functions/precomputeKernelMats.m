function [H, G, B] = precomputeKernelMats(kfncs, Dict, dataSet)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
disp('Generating precomputed kernel matrices: ');

H = cell(length(kfncs), 1);
G = cell(length(kfncs), size(dataSet, 2));
B = cell(length(kfncs), size(dataSet, 2));
parfor_progress(length(kfncs));
for kidx=1:length(kfncs)
    eta_temp = [];
    eta_temp(kidx) = 1; % place a 1 in the current kernel
    
    H{kidx,1}=computeMultiKernelMatrix(Dict,Dict,eta_temp,kfncs);
    for sidx=1:size(dataSet, 2)
        G{kidx,sidx}=computeMultiKernelMatrix(Dict,dataSet(:,sidx),eta_temp,kfncs);
        B{kidx,sidx}=computeMultiKernelMatrix(dataSet(:,sidx),dataSet(:,sidx),eta_temp,kfncs);
    end
    
    parfor_progress();
end
parfor_progress(0);
end

