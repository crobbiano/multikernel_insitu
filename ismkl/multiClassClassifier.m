function [correct_class, guessed_class, scores] = multiClassClassifier(data, irNetwork)
%*********************************************************************************************************************************
% Given a data set and an content based kernel-machine, this script evaluates the ability of the machine to correctly classify the
% samples.
%
% *** Inputs ***
% irNetwork: network structure that will be evaluated

% Contact_List: cell array of contacts whose features will used to evaluate the identification performance of the network.

% CLASS_THRESH: threshold used when assigning a class label (for results only). Compared to normalized target score.

% pool_index: Vector of indicies denoting which pools in the network to evaluate.  Sometimes only one or a few pools are changed, 
% and hence, only these pools need to be re-evaluated.  If left empty, then all pools in the network are used by default.

% pool_scores: Retrieval score produced for every pool in irNetwork by every sample in irData.  Allows us to quickly update 
% identification and classification decisions for a network that has undergone only limited change.

% *** Outputs ***
% correct_class: binary vector indicating whether or not each sample in the contact list was correctly classified.

% pool_scores: modified version of the same input variable.
%*********************************************************************************************************************************

% binary values indicating whether or not each contact was correctly classified
correct_class = zeros(length(data), 1); 

% scores for each feature vector in 'Contact_List'
scores = retrievalScore([data.features],irNetwork);

% find retrieved labels for each feature vector
numClasses = size(irNetwork.Labels,2);
[~, idxs] = max(scores,[],2);
retrieve_labels = full(ind2vec(idxs', numClasses))';

% find which features are correctly classified
gt_labels = full(ind2vec([data.gt], numClasses))';
correct_class = all(retrieve_labels == gt_labels,2);
[~,guessed_class] = max(retrieve_labels,[],2);