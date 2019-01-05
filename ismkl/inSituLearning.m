function [irNetwork, results] = inSituLearning(trainData, isData, genData, in_situ_samp, ResidNorm, SS_THRESH, CLASS_THRESH, irNetwork)
%*********************************************************************************************************************************
% Given an 'in-situ learning' contact list and an IR in-situ network, a scenario is simulated where the network can potentially
% learn up to half the patterns in the contact list, i.e. in-situ learning is performed.  After each training iteration, the
% performance of the network is evaluated in several areas to ensure the in-situ learning process has a net positive benefit on
% the network.  Included in this performance evaluation is a 'generealization' contact list, that the network has never had the
% opportunity to learn, and hence, performance on these contacts is a good indication of the system's generalization ability.
%
% *** INPUTS ***
% Test_Contacts: array of (NSWC format) contact structures that will be used for in-situ learning.
%
% Gen_Contacts: array of (NSWC format) contact structures that will be used to evaluate the performance of the IR network.
%
% in_situ_samp: array of integers denoting the order samples in Test_Contacts should be presented to the network for
% potential learning.
%
% SS_THRESH: threshold for the selective sampling measure that is used to determine whether or not a sample should be added to the
% candidate pool.
%
% CLASS_THRESH: threshold used when assigning a class label (for results only). Compared to normalized target score.
%
% DESIRED_SCORE: desired score for the input pattern relative to the pool that will learn it.
%
% net_name: name of network to load (and save after in-situ learning)
%
% *** OUTPUTS ***
% irNetwork: modified version of the starting network (has undergone in-situ learning and pool expansion).
%
% results: structure containing various performance measures stored during the learning process.
%
%
% *** SBIR DATA RIGHTS ***
% Contract No: N00014-09-M-0167 and N00014-12-C-0017
% Contractor Name: Information System Technologies, Inc.
% Contractor Address: 425 W Mulberry St., Suite 108, Fort Collins, CO 80521
% Expiration of SBIR Data Rights Period:  December 31st, 2017
%
% The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software
% marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Noncommercial
% Technical Data and Computer Software--Small Business Innovative Research (SBIR) Program clause contained in the above identified
% contract. No restrictions apply after the expiration date shown above. Any reproduction of technical data, computer software, or
% portions thereof marked with this legend must also reproduce the markings.
%*********************************************************************************************************************************
%% Instantiate Parameters Used to Track Performance
batchSize = 50;
numBatches = floor(length(isData)/batchSize);
% numBatches = floor(600/batchSize);
% numBatches = 1000/batchSize;
% numBatches = 1;
% log correct identification (on test and gen data) and classification each round
correct_class_train = zeros(length(trainData), numBatches + 1);
correct_class_is = zeros(length(isData), numBatches + 1);
correct_class_gen = zeros(length(genData), numBatches + 1);
% AUC_train = zeros(length(trainData) + 1, 1);
% AUC_is = zeros(length(isData) + 1, 1);
% AUC_gen = zeros(length(genData) + 1, 1);
sample_size = zeros(length(isData) + 1, 1);
center_size = zeros(length(isData) + 1, 2);
testing_ignored = zeros(length(isData), 1);
atoms_added = 0;

%% Evaluate Identification Performance on Fresh Network
[correct_class_train(:,1), scores_train] = multiClassClassifier(trainData, irNetwork);
[correct_class_is(:,1), scores_test] = multiClassClassifier(isData, irNetwork);
[correct_class_gen(:,1), scores_gen] = multiClassClassifier(genData, irNetwork);
disp(['Pre-In-Situ Learning CC Rate (Train Data): ' num2str(mean(correct_class_train(:,1)))])
disp(['Pre-In-Situ Learning CC Rate (IS Data): ' num2str(mean(correct_class_is(:,1)))])
disp(['Pre-In-Situ Learning CC Rate (Generalization Data): ' num2str(mean(correct_class_gen(:,1)))])

% [AUC_train(1),fa_rate_ini_train,cc_rate_ini_train] = ROCgen(irNetwork, trainData, false);
% [AUC_is(1),fa_rate_ini_is,cc_rate_ini_is] = ROCgen(irNetwork, isData, false);
% [AUC_gen(1),fa_rate_ini_gen,cc_rate_ini_gen] = ROCgen(irNetwork, genData, false);

%% *** Record size of network ***
sample_size(1) = size(irNetwork.KernMat,1);
center_size(1,:) = [nnz(irNetwork.WeightMat(:,1)) nnz(irNetwork.WeightMat(:,2))];

%% Train Network One Sample at a Time - FIXME, can do batch here
% log training samples that can still be used
available = 1:length(isData);

display(['Doing in-situ learning']);
parfor_progress(numBatches);

if isempty(in_situ_samp)
    new_in_situ_samp = zeros(length(isData), 1); % if no sample order was specified save the one that will be generated
end
% *** Learn patterns in testing set ***
for train_round = 1:numBatches
    % Select Random Sample
    if isempty(in_situ_samp) % select available sample to train in-situ
        random_sample = randperm(length(available), batchSize);
        selected_samples = available(random_sample);
        available(random_sample) = []; % remove added sample from those available to add
        new_in_situ_samp((train_round-1)*batchSize + 1:(train_round-1)*batchSize + batchSize) = selected_samples;
    else % use input sample order
        selected_samples = in_situ_samp((train_round-1)*batchSize + 1:(train_round-1)*batchSize + batchSize);
    end
    
    currData = isData(selected_samples); % contact we will use to train
    %     currData = isData((train_round-1)*batchSize + 1:(train_round-1)*batchSize + batchSize);
    
    % Potentially Adapt Network to Sample
    %     [irNetwork,update_id] = learnContact(irNetwork,currData,SS_THRESH,ResidNorm);
    [irNetwork,update_id] = learnBatch(irNetwork,currData,SS_THRESH,ResidNorm);
    
    % Evaluate Identification Performance
    if update_id > 0
        [correct_class_train(:, train_round+1), ~] = multiClassClassifier(trainData, irNetwork); % evaluate testing data
        [correct_class_is(:, train_round+1), ~] = multiClassClassifier(isData, irNetwork); % evaluate testing data
        [correct_class_gen(:, train_round+1), ~] = multiClassClassifier(genData, irNetwork); % evaluate generalization data
        
        %         AUC_train(train_round + 1) = ROCgen(irNetwork, trainData, false);
        %         AUC_is(train_round + 1) = ROCgen(irNetwork, isData, false);
        %         AUC_gen(train_round + 1) = ROCgen(irNetwork, genData, false);
        
        sample_size(train_round+1) = size(irNetwork.KernMat,1);
        center_size(train_round+1,:) = [nnz(irNetwork.WeightMat(:,1)) nnz(irNetwork.WeightMat(:,2))];
        
        if update_id == 2
            atoms_added = atoms_added + batchSize;
            %             display('Adding atoms');
        end
    else
        % results same as last iteration if no learning takes place
        correct_class_train(:, train_round+1) = correct_class_train(:, train_round);
        correct_class_is(:, train_round+1) = correct_class_is(:, train_round);
        correct_class_gen(:, train_round+1) = correct_class_gen(:, train_round);
        %         AUC_train(train_round + 1) = AUC(train_round);
        %         AUC_is(train_round + 1) = AUC(train_round);
        %         AUC_gen(train_round + 1) = AUC(train_round);
        sample_size(train_round+1) = sample_size(train_round);
        center_size(train_round+1,:) = center_size(train_round,:);
    end
    
    % show progress and canel if necessary
    %     results_str = sprintf('Train Class: %f;  IS Class: %f;  Gen Class: %f;  AUC_train: %f; AUC_is: %f AUC_gen: %f;', ...
    %         mean( correct_class_train(:, train_round+1) ), ...
    %         mean( correct_class_is(:, train_round+1) ), ...
    %         mean( correct_class_gen(:, train_round+1) ), ...
    %         AUC_train(train_round), ...
    %         AUC_is(train_round), ...
    %         AUC_gen(train_round));
    
    parfor_progress;
end
parfor_progress(0);

if isempty(in_situ_samp) % save generated sample order (if no input order was specified)
    in_situ_samp = new_in_situ_samp;
    save('in_situ_samp.mat', 'in_situ_samp');
end

% *** Compute Final ROC ***
% [~,fa_rate_fin_train,cc_rate_fin_train,~] = ROCgen(irNetwork, trainData, false);
% [~,fa_rate_fin_is,cc_rate_fin_is,~] = ROCgen(irNetwork, isData, false);
% [~,fa_rate_fin_gen,cc_rate_fin_gen,~] = ROCgen(irNetwork, genData, false);

% *****************
% Display Results *
%******************

% correct classication rate change
% figure; plot( mean(correct_class_is, 1) );
% hold on; plot( mean(correct_class_gen, 1), 'r' ); xlim([1 length(AUC)])
% xlabel('In-Situ Learning Iteration'); ylabel('Correct Classification Rate')
% legend('Testing','Generalization', 'Location', 'NorthWest')

% *** Save Results ***
results.batchSize = batchSize;
results.atoms_added = atoms_added;
results.correct_class_train = correct_class_train;
results.correct_class_is = correct_class_is;
results.correct_class_gen = correct_class_gen;
disp(['Post-In-Situ Learning CC Rate (Train Data): ' num2str(mean(results.correct_class_train(:, end)))])
disp(['Post-In-Situ Learning CC Rate (IS Data): ' num2str(mean(results.correct_class_is(:, end)))])
disp(['Post-In-Situ Learning CC Rate (Generalization Data): ' num2str(mean(results.correct_class_gen(:, end)))])

% AUC change
% figure; plot(AUC); xlim([1 length(AUC)])
% xlabel('In-Situ Learning Iteration'); ylabel('AUC')
%
% figure;
% plot(fa_rate_ini,cc_rate_ini,'LineWidth',2);
% hold on;
% plot(fa_rate_fin,cc_rate_fin,'r','LineWidth',2)
% xlabel('P_{FA}')
% ylabel('P_{CC}')
% legend('After Baseline Training','After In-Situ Learning')
% grid on

% % final network size/growth
% figure; stem(pool_size(2, :) - pool_size(1, :));
% xlabel('Pool Index'); ylabel('Number of Neurons Added')
% xlim([0.5 (size(pool_size, 2) + 0.5)])
%
% % ROC for final network
% [~, fa_rate, cc_rate] = ROCgen(irNetwork, Gen_Contacts, true); % hold on; plot(0.04167, 1 - 0.04167, '*')
% xlabel('P_{FA}'); ylabel('P_{CC}')
%
% disp(['Proportion of testing samples ignored: ' num2str(mean(testing_ignored))])


% results.AUC_gen = AUC;
% results.fa_rate_gen = fa_rate;
% results.cc_rate_gen = cc_rate;

% results.pre_pool_size = pool_size(1, :);
% results.post_pool_size = pool_size(2, :);
% results.net_size = net_size;
