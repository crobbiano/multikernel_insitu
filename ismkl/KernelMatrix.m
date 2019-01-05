function Kmat = KernelMatrix(X,Y,Params,Kernels)

% Compute matrix containing all pair-wise distances between data matrices X and Y
DistMat = pdist2(X.',Y.');
[M,N] = size(DistMat);
K = length(Params);

KmatTemp = zeros(M,N*K);
for k = 1:length(Params)
    kernel = Kernels{k};
    switch lower(kernel)
       
        case 'quartic' % Implement Quartic Kernel
            tmpMat = (1-DistMat.^2./(2*Params(1,k)^2)).^2;
            tmpMat(DistMat.^2 >= 2*Params(1, k)^2) = 0;  
            KmatTemp(:,(k-1)*N+1:k*N) = tmpMat;
        case 'gaussian'
            KmatTemp(:,(k-1)*N+1:k*N) = exp(-DistMat.^2./(2*Params(1,k)^2));
        case 'poly'
            KmatTemp(:,(k-1)*N+1:k*N) = (X.'*Y + Params(1,k)).^Params(2,k);
        case 'linear'
            KmatTemp(:,(k-1)*N+1:k*N) = (X.'*Y + Params(1,k));
        case 'tanh'
            KmatTemp(:,(k-1)*N+1:k*N) = tanh(Params(1,k) + Params(2,k)*(X.'*Y));
        otherwise
            error('Unrecognized Kernel Type')
    end
end

Kmat = zeros(M,N*K);
for i = 1:N
    Kmat(:,(i-1)*K+1:i*K) = KmatTemp(:,i:N:N*K);
end