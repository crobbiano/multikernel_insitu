function [X, h, g, z, zm, c, cc_rate, fa_rate, poor_idxs] = mklsrcUpdateWithAddition(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class, baseline, trainSetSmall, errorGoal)
%mklsrcUpdate Do an update for MKL-SRC
%   Detailed explanation goes here

optionKSRSC.lambda=0.1;
% optionKSRSC.SCMethod='l1qpAS'; % can be nnqpAS, l1qpAS, nnqpIP, l1qpIP, l1qpPX, nnqpSMO, l1qpSMO
optionKSRSC.iter=1500;
optionKSRSC.dis=0;
optionKSRSC.residual=1e-4;
optionKSRSC.tof=1e-4;

H=computeMultiKernelMatrixFromPrecomputed(Hfull,eta);
retrieve_score = zeros(length(trainClassSmall),length(classes));
h = zeros(1,length(trainClassSmall));
z = zeros(1,length(trainClassSmall));

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
%         X(:,idx) = OMP(Htmp, G, 35, errorGoal);
    X(:,idx) = RecursiveOMP(Htmp, [], G, errorGoal);
    % Find class - calculate h (class) and z (correct class)
    classerr = zeros(1, length(classes));
    for class=1:length(classes)
        b_idx = sum(num_per_class(1:class-1)) + 1;
        e_idx = sum(num_per_class(1:class));
        x_c = X(b_idx:e_idx ,idx);
        %         x2_c = X2(b_idx:e_idx ,idx);
        kernel_c = H(b_idx:e_idx ,b_idx:e_idx);
        partial_c = G(b_idx:e_idx)';
        classerr(class) = B + x_c'*kernel_c*x_c - 2*partial_c*x_c;
        %         classerr2(class) = B + x2_c'*kernel_c*x2_c - 2*partial_c*x2_c;
    end
    retrieve_score(idx, :) = classerr;
    [~, h(idx)] = min(classerr);
    h(idx)=classes(h(idx));
    z(idx) = (h(idx) == trainClassSmall(idx));
    
    % Need to calculate the ability to classify for each individual kernel
    for kidx=1:length(eta)
        eta_temp = zeros(size(eta));
        eta_temp(kidx) = 1; % place a 1 in the current kernel
        
        H_temp = computeMultiKernelMatrixFromPrecomputed(Hfull, eta_temp);
        G_temp = computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta_temp);
        B_temp = computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta_temp);
        
        err_temp = zeros(1, length(classes));
        for class=1:length(classes)
            b_idx = sum(num_per_class(1:class-1)) + 1;
            e_idx = sum(num_per_class(1:class));
            x_c = X(b_idx:e_idx ,idx);
            kernel_c = H_temp(b_idx:e_idx ,b_idx:e_idx);
            partial_c = G_temp(b_idx:e_idx)';
            err_temp(class) = B_temp + x_c'*kernel_c*x_c - 2*partial_c*x_c;
        end
        [~, h_temp] = min(err_temp);
        h_temp = classes(h_temp);
        g(kidx, idx) = h_temp;
        zm(kidx, idx) = g(kidx, idx) == trainClassSmall(idx);
        
        if sum(1-z)
            c(kidx,1) = sum(zm(kidx, find(z(1:idx)==0)))/sum(1-z);
        else
            c(kidx,1) = 0;
        end
        
        
    end
    
end


% Find components that have an error greater then a particular threshold OR
% the difference between reconstruction error between classes is greater
% than another threshold
% poor_confidence_idxs = min(retrieve_score,[],2) > .97;
llr = log10(max(retrieve_score,[],2) ./ min(retrieve_score,[],2));
% poor_differentiation_idxs = abs(llr) < .02;
poor_differentiation_idxs = abs(llr) < .5;
% poor_differentiation_idxs = max(abs(retrieve_score - min(retrieve_score,[],2)),[],2) < max(max(abs(retrieve_score - min(retrieve_score,[],2)),[],2))/2;
% llr = abs(retrieve_score(:,1) - retrieve_score(:,2));
% poor_differentiation_idxs = llr < .2;

% llr = max(retrieve_score,[],2) - min(retrieve_score,[],2);
% poor_differentiation_idxs = llr < .1;
% poor_idxs = poor_confidence_idxs & poor_differentiation_idxs;
poor_idxs = poor_differentiation_idxs;

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
end

