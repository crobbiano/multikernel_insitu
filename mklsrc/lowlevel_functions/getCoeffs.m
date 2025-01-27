function [x] = getCoeffs(prev_x, y, A, kernels, kappa, lambda, eta, num_iter, train_idx)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    
    % compute the current kernel based on weights
    curr_kernel = zeros(size(A,2));
    % Some tom-foolry with the partial kernel to make it work with under lying operations
    % stick the sample, y, in the first column of a matrix of size A and zero out everying
    % else.  then after the computation is done, extract the first column again
    partial_kernel = zeros(size(A,2));
%     partial_kernel = zeros(1, size(A,2));
    single_kernel = 0;
    
    y_mat = zeros(size(A));
    y_mat(:,1) = y;
    
    for i=1:length(kappa)
        option.kernel = 'cust'; option.kernelfnc=kappa{i};
        if train_idx == 0
            curr_kernel = curr_kernel + eta(i)*computeKernelMatrix(A,A,option);
            partial_kernel = partial_kernel + eta(i)*computeKernelMatrix(y_mat,A,option);
            single_kernel = single_kernel + eta(i)*computeKernelMatrix(y,y,option);
        else 
            curr_kernel = curr_kernel + eta(i)*kernels{i};
            partial_kernel = partial_kernel + eta(i)*kernels{i}(train_idx, :);
            single_kernel = single_kernel + eta(i)*kernels{i}(train_idx,train_idx);
        end
    end
    
    partial_kernel = partial_kernel(1,:);
 
    
    if train_idx > 0
        partial_kernel(:, train_idx) = 0;
        curr_kernel(:, train_idx) = 0;
        curr_kernel(train_idx, :) = 0;
    end
    
    option.lambda = lambda;
    option.iter = num_iter;
%     option.SCMethod='l1qpAS';
%     option.SCMethod='l1nnlsSMO';
    x = KSRSC(curr_kernel, partial_kernel', single_kernel, option);
end

