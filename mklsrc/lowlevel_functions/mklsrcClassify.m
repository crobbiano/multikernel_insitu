function [X, h, g, z, zm, c, cc_rate, fa_rate] = mklsrcClassify(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class, baseline, errorGoal)
%mklsrcUpdate Do an update for MKL-SRC
%   Detailed explanation goes here

optionKSRSC.lambda=0.1;
% optionKSRSC.SCMethod='l1qpAS'; % can be nnqpAS, l1qpAS, nnqpIP, l1qpIP, l1qpPX, nnqpSMO, l1qpSMO
optionKSRSC.iter=200;
optionKSRSC.dis=0;
optionKSRSC.residual=1e-4;
optionKSRSC.tof=1e-4;


H=computeMultiKernelMatrixFromPrecomputed(Hfull,eta);

trainLen = length(trainClassSmall);

retrieve_score = zeros(trainLen, length(classes));
h = zeros(1, trainLen);
z = zeros(1, trainLen);


% parfor_progress(length(trainClassSmall));
for idx = 1:length(trainClassSmall)
    % compute kernels
    G=computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta);
    B=computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta);
    
    if baseline
        G(idx) = 0;
        Htmp = H;
        Htmp(:, idx) = 0;
        Htmp(idx, :) = 0;
        %         for i=1:length(Hfull)
        %             Htmp2tmp = Hfull{i};
        %             Htmp2tmp(:, idx) = 0;
        %             Htmp2tmp(idx, :) = 0;
        %             Htmp2 = horzcat(Htmp2, Htmp2tmp);
        %         end
    else
        Htmp = H;
        %         Htmp2 = horzcat([Hfull{:}]);
    end
    % KSRSC sparse coding
    %         [X(:, idx), ~, ~] =KSRSC(H,G,diag(B),optionKSRSC);
    %     X(:,idx) = OMP(Htmp, G, 35, errorGoal);
    X = RecursiveOMP(Htmp, [], G, errorGoal);
    
    % Find class - calculate h (class) and z (correct class)
    classerr = zeros(1, length(classes));
    for class=1:length(classes)
        b_idx = sum(num_per_class(1:class-1)) + 1;
        e_idx = sum(num_per_class(1:class));
        x_c = X(b_idx:e_idx);
        kernel_c = H(b_idx:e_idx ,b_idx:e_idx);
        partial_c = G(b_idx:e_idx)';
        classerr(class) = B + x_c'*kernel_c*x_c - 2*partial_c*x_c;
    end
    retrieve_score(idx, :) = classerr;
    [~, h(idx)] = min(classerr);
    h(idx)=classes(h(idx));
    z(idx) = (h(idx) == trainClassSmall(idx));
    
%     parfor_progress;
end

thresh = linspace(min(min(retrieve_score))-.01,max(max(retrieve_score))+.01,5000);
cc_rate = zeros(length(classes), length(thresh));
fa_rate = zeros(length(classes), length(thresh));
for label_idx=1:length(classes)
    for i = 1:length(thresh)
        cc_rate(label_idx, i) = length(find(retrieve_score(logical(trainClassSmall==classes(label_idx)),label_idx)<=thresh(i)))/...
            length(retrieve_score(logical(trainClassSmall==classes(label_idx))));
        fa_rate(label_idx, i) = length(find(retrieve_score(~logical(trainClassSmall==classes(label_idx)),label_idx)<=thresh(i)))/...
            length(retrieve_score(~logical(trainClassSmall==classes(label_idx))));
    end
end
% parfor_progress(0);
X = 0;
g = 0;
zm = 0;
c = 0;
end

