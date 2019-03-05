for runNum = 1:1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This runs the series of scripts needed to perform ISMKL, including
    % loading the data, training the base network, and performing in-situ
    % learning then classification.
    clearvars -except runNum
    close all
    clc
    
    %% Define Parameters
    netName = 'irNetwork';
    
    baseNetworkFile = './irNetwork_base.mat';
    isNetworkFile = './irNetwork_is.mat';
%     allDataPath = '../gen_scripts/yale.mat';
%     allDataPath = '../gen_scripts/yale_dark.mat';
    allDataPath = '../gen_scripts/davis_base_boss_is.mat';
    
    ResidNorm = .15;
    SS_THRESH = 15;
    CLASS_THRESH = 0.25;
    
    shift = 0;    
    %% Load data
    [trainData, isData, genData] = loadAllData(allDataPath, shift);
%     [isData,trainData,genData] = loadAllData(allDataPath, shift);
    %% Train Baseline Classifier
    trainNewBase = 1;
    if trainNewBase
        baseNetwork = trnMultikernel([trainData.gt],[trainData.features]);
        saveBase = 1;
        if saveBase
            save(baseNetworkFile, 'baseNetwork');
        end
        
        Count = zeros(length(baseNetwork.Params),1);
        for i = 1:length(baseNetwork.Params)
            Count(i) = nnz(baseNetwork.WeightMat(i:size(baseNetwork.Params,1):size(baseNetwork.WeightMat,1),:));
        end
        figure;
        stem(baseNetwork.Params(1,:),Count,'LineWidth',2)
        xlabel('Kernel Parameter (\sigma)')
        ylabel('Number of Kernels Selected')
        axis([0 baseNetwork.Params(end)+1 0 max(Count)+1])
        grid on
        
    else
        load(baseNetworkFile);
    end
    
    %% Perform In-Situ Learning
    [isNetwork, results] = inSituLearning(trainData, isData, genData, [], ResidNorm, SS_THRESH, CLASS_THRESH, baseNetwork);
    
    %% Save stuff
    saveIS = 1;
    if saveIS
        save(isNetworkFile, 'isNetwork');
    end
    save(['results/results_batch' num2str(results.batchSize) '_run' num2str(runNum) 'multi.mat'], 'results', '-v7.3');
end


%% Plot some things

figure(898); clf; hold on
plot(mean(results.correct_class_train, 1),'-.')
plot(mean(results.correct_class_is, 1),'--')
plot(mean(results.correct_class_gen, 1))
legend('Baseline', 'In-Situ', 'Generalization', 'Location', 'southwest')
ylim([.5, 1])
xlim([1 size(results.correct_class_gen,2)])
xxticks = xticks*results.batchSize;
xticklabels(xxticks)
xlabel('In-Situ samples learned')
title(['Atoms added: ' num2str(results.atoms_added)])
ylabel('Correct Classification Rate')