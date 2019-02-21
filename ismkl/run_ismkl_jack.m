
single_kernel=0;
experiment_no=3;
for runNum = 1:1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This runs the series of scripts needed to perform ISMKL, including
    % loading the data, training the base network, and performing in-situ
    % learning then classification.
    clearvars -except runNum single_kernel experiment_no
    close all
    clc
    
    %% Define Parameters
    netName = 'irNetwork';
    
    baseNetworkFile = './irNetwork_base.mat';
    isNetworkFile = './irNetwork_is.mat';
    
    %
    
    switch experiment_no
        case 1
            allDataPath='../gen_scripts/frm_trex_pond_insitu_2class_test_insitu0.1.mat';  % %Dataset for Experiment 1
        case 2
            allDataPath ='../gen_scripts/frm_trex_pond_insitu_2class_test_insitu_single_gen0.1.mat'; %Dataset for Experiment 2
        case 3
            allDataPath ='../gen_scripts/frm_trex_pond_insitu_2class_test_insitu_single_gen_ol_0.1.mat'; %Dataset for Experiment 3
        otherwise
            allDataPath='../gen_scripts/frm_trex_pond_insitu_2class_test_insitu0.1.mat';  % %Dataset for Experiment 1
    end
    %
    
    %allDataPath = '../gen_scripts/trex_pond_insitu_2class_test_insitu.mat';
    % allDataPath = '../gen_scripts/trex_insitu_multi_test_1000insitu.mat';
    
    ResidNorm = 0.1;%.26;%
    SS_THRESH = 50;
    CLASS_THRESH = 0.25;
    
    %% Load data
    %[trainData, isData,isData2, genData] = loadAllData_jack(allDataPath);
    [trainData, isData1, isData2, genData1, genData2] = loadAllData_jack_full(allDataPath);
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
    
    if single_kernel
        
    end
    
    %% Perform In-Situ Learning
    %SS_THRESH=floor(length(isData1)/20/5);
    [isNetwork, results]   = inSituLearning(trainData, isData1, genData1, [], ResidNorm, SS_THRESH, CLASS_THRESH, baseNetwork);
    %trainData2=[trainData(:);isData1(:)]
    %SS_THRESH=floor(length(isData2)/20/5);
    [isNetwork2, results2] = inSituLearning(trainData, isData2, genData2, [], ResidNorm, SS_THRESH, CLASS_THRESH, isNetwork);
    
    %% Save stuff
    saveIS = 1;
    if saveIS
        save(isNetworkFile, 'isNetwork');
    end
    save(['results/results_batch' num2str(results.batchSize) '_run' num2str(runNum) '.mat'], 'results');
end


%% Plot some things

figure(898); clf; hold on
plt1=plot(mean(results.correct_class_train, 1),'-.');
plt2=plot(mean(results.correct_class_is, 1),'--');
plt3=plot(mean(results.correct_class_gen, 1));

[~,i1max]= max(plt1.YData);
[~,i2max]= max(plt2.YData);
[~,i3max]= max(plt3.YData);
% makedatatip(plt1,[size(results.correct_class_train,2) i1max ]);
% makedatatip(plt2,[size(results.correct_class_is,2) i2max]);
% makedatatip(plt3,[size(results.correct_class_gen,2) i3max]);

legend('Baseline', 'In-Situ', 'Generalization','location','southeast')
ylim([0, 1])
title(['First IS - Atoms Added: ' num2str(results.atoms_added)])
xlabel('Training Batch #');
ylabel('Correct Classification Rate');
grid minor

ylim([.4 1])
figure(899); clf; hold on

plt1=plot(mean(results2.correct_class_train, 1),'-.');
plt2=plot(mean(results2.correct_class_is, 1),'--');
plt3=plot(mean(results2.correct_class_gen, 1));

[~,i1max]= max(plt1.YData);
[~,i2max]= max(plt2.YData);
[~,i3max]= max(plt3.YData);
% makedatatip(plt1,[size(results2.correct_class_train,2) i1max ]);
% makedatatip(plt2,[size(results2.correct_class_is,2) i2max]);
% makedatatip(plt3,[size(results2.correct_class_gen,2) i3max]);

legend('Baseline', 'In-Situ', 'Generalization','location','southeast')
ylim([0, 1])
title(['Second IS - Atoms Added: ' num2str(results2.atoms_added)])
xlabel('Training Batch #');
ylabel('Correct Classification Rate');
grid minor
ylim([.4 1])