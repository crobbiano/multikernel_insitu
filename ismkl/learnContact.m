function [irNetwork, update_id] = learnContact(irNetwork, data, SS_THRESH, ResidNorm)
%*********************************************************************************************************************************
% Adapts an IR in-situ learning network to a single pattern (contact).

% *** INPUTS ***
% ClassStruct: structure array containing variables used by the classifier

% Contact: NSWC format structure containing the information about a single image snippet. The network will potentially learn the
% features associated with this contact.

% CONF_THRESH: threshold for how confident the classifier labels the contact

% SS_THRESH: threshold for the maximum increase in the number of coefficients allowed to enter the model

% ResidNorm: maximum residual norm used by OMP algorithm

% *** OUTPUTS ***
% ClassStruct: (potentially) modified version of the input classification structure where the weights of the classifier may have
% been updated to account for the input feature vector

% update_id: integer indicating if and what update was applied to classifier
%            0 = classifier correctly labels input contact with sufficient confidence, no update required
%            1 = update in classifier brings small sacrifice in sparsity, update weights
%            2 = update in classifier brings significant sacrifice in sparsity, update weights and add kernel centers

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
%% Submit Sample and Retrieve Score  
score = retrievalScore(data.features,irNetwork); % Compute score
numClasses = size(irNetwork.Labels,2);
[~,class_ind] = max(score); % Find class with highest score
class_label = full(ind2vec(class_ind, numClasses));
gt_label = full(ind2vec([data.gt], numClasses))';

%% Possibly Update Classifier - FIXME, can do batch here
if all(class_label == gt_label) % if correctly labeled with high confidence
    % Report results of update
    update_id = 0;    
else % try updating weights and checking increase in sparsity
    % Compute kernel matrix for input feature vector and add
    K10 = KernelMatrix(data.features,irNetwork.Features,irNetwork.Params,irNetwork.Kernels);
    Kmat = [irNetwork.KernMat;K10];
    
    % Add new label
    L = [irNetwork.Labels;gt_label];
    
    % Compute new weight matrix
    W = RecursiveOMP(Kmat,irNetwork.WeightMat,L,ResidNorm);
    
    % Find increase in number of coefficients 
    DelNumCoefficients = max(sum( W ~= 0, 1 ) - sum( irNetwork.WeightMat ~= 0, 1 ));
    
    if DelNumCoefficients < SS_THRESH % if number of coefficients added small, update classifier
        % Update classifier
        irNetwork.TrainData = [irNetwork.TrainData,data.features];
        irNetwork.KernMat = Kmat;
        irNetwork.WeightMat = W;
        irNetwork.Labels = L;
        
        % Report results of update
        update_id = 1;
    else % add kernel centers
        % Compute new kernel matrix for input feature vector
        K01 = KernelMatrix(irNetwork.TrainData,data.features,irNetwork.Params,irNetwork.Kernels);
        K11 = KernelMatrix(data.features,data.features,irNetwork.Params,irNetwork.Kernels);
        Kmat = [irNetwork.KernMat,K01;K10,K11];
        
        % Compute new weight matrix
        W = RecursiveOMP(Kmat,irNetwork.WeightMat,L,ResidNorm);
        
        % Update classifier
        irNetwork.Features = [irNetwork.Features,data.features];
        irNetwork.TrainData = [irNetwork.TrainData,data.features];
        irNetwork.KernMat = Kmat;
        irNetwork.WeightMat = W;
        irNetwork.Labels = L;
        
        % Report results of update
        update_id = 2;
    end
    
end