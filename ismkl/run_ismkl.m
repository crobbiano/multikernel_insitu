for runNum = 1:25
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This runs the series of scripts needed to perform ISMKL, including
    % loading the data, training the base network, and performing in-situ
    % learning then classification.
    clearvars -except runNum
    %     close all
    %     clc
    
    %% Define Parameters
    netName = 'irNetwork';
    
    baseNetworkFile = './irNetwork_base.mat';
    isNetworkFile = './irNetwork_is.mat';
    %     allDataPath = '../gen_scripts/yale.mat';
    %     allDataPath = '../gen_scripts/yale_dark.mat';
    allDataPath = '../gen_scripts/yale_darkIS_darkMediumGen.mat';
    %     allDataPath = '../gen_scripts/davis_base_boss_is.mat';
    %     allDataPath = '../gen_scripts/boss_base_davis_is.mat';
    
    ResidNorm = .3;
    SS_THRESH = 13;
    CLASS_THRESH = 0.25;
    
    shift = 0;
    %% Load data
    [trainData, isData, genData] = loadAllData(allDataPath, shift);
    %     [isData,trainData,genData] = loadAllData(allDataPath, shift);
    %% Train Baseline Classifier
    trainNewBase = 0;
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
        %         figure;
        %         stem(baseNetwork.Params(1,:),Count,'LineWidth',2)
        %         xlabel('Kernel Parameter (\sigma)')
        %         ylabel('Number of Kernels Selected')
        %         axis([0 baseNetwork.Params(end)+1 0 max(Count)+1])
        %         grid on
        
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
    save(['results/results_batch' num2str(results.batchSize) '_run' num2str(runNum) 'multi_5batch.mat'], 'results', '-v7.3');
    
    %% Plot some things
    
    figure(898); clf; hold on
    plot(mean(results.correct_class_train, 1),'-.')
    plot(mean(results.correct_class_is, 1),'--')
    plot(mean(results.correct_class_gen, 1))
    legend('Baseline', 'Incremental Learning in Environment 1', 'Generalization', 'Location', 'southwest')
    ylim([.5, 1])
    xlim([1 size(results.correct_class_gen,2)])
    xxticks = xticks*results.batchSize;
    xticklabels(xxticks)
    xlabel('Incremental samples learned')
    title(['Atoms added: ' num2str(results.atoms_added)])
    ylabel('Correct Classification Rate')
    
    %% generate confusion matrices
    % temp
    plotConf = 0;
    if plotConf
        for i=1:25
            load(['C:\Users\iamchris\Documents\multikernel_journal\ismkl\results\results_batch50_run' num2str(i) 'multi_conf.mat'])
            
            I=eye(31);
            figure(1)
            [~, cm_base, ~, ~] = confusion(num2bin10([trainData.gt], length(unique([trainData.gt]))), num2bin10(results.guess_class_train, length(unique([trainData.gt]))));
            imagesc(normc(cm_base));
            % colormap((gray))
            colorbar
            % title('Confusion matrix for baseline data')
            figure(2)
            [~, cm_test, ~, ~] = confusion(num2bin10([isData.gt], length(unique([isData.gt]))), num2bin10(results.guess_class_is, length(unique([isData.gt]))));
            imagesc(normc(cm_test));
            % colormap((gray))
            colorbar
            % title('Confusion matrix for in-situ data')
            figure(3)
            [~, cm_valid, ~, ~] = confusion(num2bin10([genData.gt], length(unique([genData.gt]))), num2bin10(results.guess_class_gen, length(unique([genData.gt]))));
            imagesc(normc(cm_valid));
            % colormap((gray))
            colorbar
            % title('Confusion matrix for generalization data')
            
            m(i) = trace(I*cm_valid)/(norm(I, 'fro')* norm(cm_valid,'fro'))
            aa_added(i) = results.atoms_added;
        end
    end
end