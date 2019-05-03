function [irNetwork] = trnMultikernel(labels, features)
%*********************************************************************************************************************************
% Initializes and baseline trains a multikernel classifier based on the input features and object type labels

% *** INPUTS ***
% labels: 1 x N cell array containing ground truth information (class) for N contacts.

% features: L x N matrix containing the feature vectors of these contacts as columns

% data_fname: name of the new classifier data file

% *** OUTPUTS ***
% None.  Network structure is save to data_fname


% *** SBIR DATA RIGHTS ***
% Contract No: N00014-09-M-0167 and N00014-12-C-0017
% Contractor Name: Information System Technologies, Inc.
% Contractor Address: 425 W Mulberry St., Suite 108, Fort Collins, CO 80521
% Expiration of SBIR Data Rights Period:  December 31st, 2017

% The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software 
% marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Noncommercial 
% Technical Data and Computer Software--Small Business Innovative Research (SBIR) Program clause contained in the above identified
% contract. No restrictions apply after the expiration date shown above. Any reproduction of technical data, computer software, or
% portions thereof marked with this legend must also reproduce the markings.
%*********************************************************************************************************************************

%% Set Parameters 
ResidNorm = 0.1; % norm of residual used in OMP
 % kernel parameters used by the classifier, allow for multi params
GaussParams = [.5*linspace(.01,3,10);zeros(1,10)];
PolyParams = [.5 .5 .5 .5 1 1 1 1; 1 2 3 4 1 2 3 4];
% PolyParams = [];
Params = [GaussParams, PolyParams];

Kernels = cell(0); % text variable identifying the type of kernel
for k = length(Kernels)+1:length(Kernels)+size(GaussParams,2)
    Kernels{k} = 'gaussian';
end
for k = length(Kernels)+1:length(Kernels)+size(PolyParams,2)
    Kernels{k} = 'poly';
end

%% Baseline Train Classifier 
% Define label matrix
L0 = full(ind2vec(labels))';

% Build kernel matrix
K00 = KernelMatrix(features,features,Params,Kernels);

% Solve for weight matrix
W0 = RecursiveOMP(K00,[],L0,ResidNorm);

% Build classifier structure
irNetwork.Features = features;
irNetwork.TrainData = features;
irNetwork.KernMat = K00;
irNetwork.WeightMat = W0;
irNetwork.Labels = L0;
irNetwork.Params = Params;
irNetwork.Kernels = Kernels;

% *** Save network and feature normalization parameters ***
% save(data_fname, 'irNetwork')