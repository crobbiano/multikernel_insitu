function scores = retrievalScore(data_features, irNetwork)
%*********************************************************************************************************************************
% Takes a query feature vector and a(n) (image) retrieval network and finds the scores the query produces for each of the pools 
% in the network. These scores (and the indices of the pools that produced them) can be used to determine those features in the 
% network that are most similar to the query.


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

% Compute Kernel Matrix For All Query Features
K = KernelMatrix(data_features,irNetwork.Features,irNetwork.Params,irNetwork.Kernels);

% Compute Scores For Every Query Feature
scores = K*sparse(irNetwork.WeightMat); 