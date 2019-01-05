clear all
clc
close all
%%
load('../TREX/YFRM_scaled_40.mat');
load('../TREX/TREX_DATA_1600_10_6.mat');

% Extracting Range and class labels from YFRM and Ytrex and removing
% mean/ truncating to appropriate range bins
low_f=30;  % 3 kHz (lowest freq of interest)
high_f=300;% 30kHz (highest freq of interest

Y_TREX=Y;
%testing
classes_TREX=Y_TREX(end-2,:); % 3rd to last row indicates class label of object in sample
range_TREX =Y_TREX(end-1,:); % 2nd to last row indicates range of object in sample
angle_TREX =Y_TREX(end,:);   % last row indicates aspect angle of sample
Y_TREX=Y_TREX(low_f:high_f,:); % Selecting all samples at our frequency bins
% of interest (1:310 <-> 0-31kHz)

bunkTREX = (classes_TREX == 1) | (classes_TREX == 8);
classes_TREX(bunkTREX) = [];
Y_TREX(:,bunkTREX) = [];

%training
classes_FRM=YFRM(end-1,:); % 2nd to last row indicates class label of object in sample
range_FRM=YFRM(end,:);   % last row indicates range of object in sample
Y_FRM=YFRM(low_f:high_f,:);


bunkFRM = (classes_FRM == 1) | (classes_FRM == 8);
classes_FRM(bunkFRM) = [];
Y_FRM(:,bunkFRM) = [];

% get rid of low power things
keep_idx = vecnorm(Y_TREX,2,1) >= 1e-5;
Y_TREX = Y_TREX(:,keep_idx);
classes_TREX = classes_TREX(keep_idx);

two_class = 1;
if two_class
    nonuxo_TREX = (classes_TREX == 2) | (classes_TREX == 3);
    uxo_TREX = (classes_TREX == 4) | (classes_TREX == 5) | (classes_TREX == 6) | (classes_TREX == 7) | (classes_TREX == 9) | (classes_TREX == 10);
%     nonuxo_TREX = (classes_TREX == 2); %| (classes_TREX == 3);
%     uxo_TREX = (classes_TREX == 5); %| (classes_TREX == 6);
    classes_TREX(nonuxo_TREX) = 0;
    classes_TREX(uxo_TREX) = 1;
    classes_TREX(~(uxo_TREX | nonuxo_TREX)) = [];
    Y_TREX(:, ~(uxo_TREX | nonuxo_TREX)) = [];
    
    
    nonuxo_FRM = (classes_FRM == 2) | (classes_FRM == 3);
    uxo_FRM = (classes_FRM == 4) | (classes_FRM == 5) | (classes_FRM == 6) | (classes_FRM == 7) | (classes_FRM == 9) | (classes_FRM == 10);
%     nonuxo_FRM = (classes_FRM == 2);% | (classes_FRM == 3);
%     uxo_FRM = (classes_FRM == 5);% | (classes_FRM == 6);
    classes_FRM(nonuxo_FRM) = 0;
    classes_FRM(uxo_FRM) = 1;
    classes_FRM(~(uxo_FRM | nonuxo_FRM)) = [];
    Y_FRM(:, ~(uxo_FRM | nonuxo_FRM)) = [];
end

% Hacking. can we even identify from the same dataset?
hackkkk = 0;
if hackkkk
    
    classes = unique(classes_TREX);
    nObs = length(classes_TREX);
    nClasses = length(classes);
    for ii = 1:nClasses
        num_per_class_train(ii) = sum(classes_TREX == classes(ii));
    end
    nSamples = min(num_per_class_train)/2;
    for ii = 1:nClasses
        idx = classes_TREX == classes(ii);
        idxs((ii-1)*nSamples+1:ii*nSamples) = randsample(classes_TREX(idx), nSamples);
        
        sampleidx = find(classes_TREX == classes(ii));
        indDict((ii-1)*nSamples+1:ii*nSamples) = randsample(sampleidx, nSamples);
    end
    
    
    numtotalvec = 1:length(classes_TREX);
    remainingindices = setdiff(numtotalvec, indDict);
    indDictSmall = indDict;
    
    Y_FRM = Y_TREX(:, indDictSmall);
    classes_FRM = classes_TREX(indDictSmall);
    Y_TREX = Y_TREX(:, remainingindices);
    classes_TREX = classes_TREX(remainingindices);
    
%     Y_FRM = Y_TREX(:, 1:round(length(Y_TREX)/2)-1);
%     classes_FRM = classes_TREX(1:round(length(classes_TREX)/2)-1);
%     Y_TREX = Y_TREX(:, round(length(Y_TREX)/2): end);
%     classes_TREX = classes_TREX(round(length(classes_TREX)/2): end);
end


shift_classes = 0;
if shift_classes
    classes_FRM = classes_FRM + 1;
    classes_TREX = classes_TREX + 1;
end

classes_train = classes_FRM;
classes_test = classes_TREX;
Data_train = Y_FRM;
Data_test = Y_TREX;

do_kmeans = 1;
if do_kmeans
    num_clusters = 600;
    Data_train_0 = Data_train(:,(classes_FRM == 0 + shift_classes));
    Data_train_1 = Data_train(:,(classes_FRM == 1 + shift_classes));
    [cidx_1,cmeans_1] = kmeans(Data_train_1',num_clusters,'dist','sqeuclidean');
    [cidx_0,cmeans_0] = kmeans(Data_train_0',num_clusters,'dist','sqeuclidean');
    classes_train = [(0 + shift_classes)*ones(1, size(cmeans_0,1)), (1 + shift_classes)*ones(1, size(cmeans_1,1))];
    Data_train = [cmeans_0; cmeans_1]';
end

allClass = classes_train;
allSet = Data_train;
allSet = normc(allSet);

%% Training sets
classes = unique(allClass);
nObs = length(allClass);
nClasses = length(classes);
for ii = 1:nClasses
    num_per_class_train(ii) = sum(allClass == classes(ii));
end
nSamples = round(min(num_per_class_train)/1);
% nSamples = 100;

subsample = 0;
if subsample
    for ii = 1:nClasses
        idx = allClass == classes(ii);
        idxs((ii-1)*nSamples+1:ii*nSamples) = randsample(allClass(idx), nSamples);
        
        sampleidx = find(allClass == classes(ii));
        indDict((ii-1)*nSamples+1:ii*nSamples) = randsample(sampleidx, nSamples);
    end

    
    numtotalvec = 1:length(allClass);
    remainingindices = setdiff(numtotalvec, indDict);
    indDictSmall = indDict;
    
    dictClass = allClass(indDict);
    dictSet=allSet(:,indDict);
    dictClassSmall=allClass(indDictSmall);
    dictSetSmall=allSet(:,indDictSmall);
else
    dictClass = allClass;
    dictSet=allSet;
    dictClassSmall=allClass;
    dictSetSmall=allSet;
end
%% Testing sets
% allClass = allClass(remainingindices);
% allSet = allSet(:,remainingindices);
allClass = classes_test;
allSet = Data_test;
allSet = normc(allSet);


for ii = 1:nClasses
    num_per_class_test(ii) = sum(allClass == classes(ii));
end

nClasses = length(classes);
% nSamples = 1000;
nSamples = round(min(num_per_class_test)/1);
test_subsample = 1;
clear indDict
if test_subsample
    for ii = 1:nClasses
        idx = allClass == classes(ii);
        idxs((ii-1)*nSamples+1:ii*nSamples) = randsample(allClass(idx), nSamples);
        
        sampleidx = find(allClass == classes(ii));
        indDict((ii-1)*nSamples+1:ii*nSamples) = randsample(sampleidx, nSamples);
    end

    
    numtotalvec = 1:length(allClass);
    remainingindices = setdiff(numtotalvec, indDict);
    indDictSmall = indDict;
    
    allClass = allClass(indDict);
    allSet=allSet(:,indDict);
end
%%

kfold=100; % Number of classes.
ind=crossvalind('Kfold',allClass,kfold);
%%
indTrain=logical(sum(ind==1:2,2));
indTrainSmall=logical(sum(ind==1:2,2));
% indTrainSmall=indTrain;

indTest=logical(sum(ind==1:20,2));
indTestSmall=logical(sum(ind==1:20,2));
% indTestSmall=indTest;

indValid=logical(sum(ind==50:100,2));
indValidSmall=logical(sum(ind==50:100,2));
% indValidSmall=indValid;

trainClass = allClass(indTrain);
trainSet=allSet(:,indTrain);
trainClassSmall=allClass(indTrainSmall);
trainSetSmall=allSet(:,indTrainSmall);

testClass = allClass(indTest);
testSet=allSet(:,indTest);
testClassSmall=allClass(indTestSmall);
testSetSmall=allSet(:,indTestSmall);

validClass = allClass(indValid);
validSet=allSet(:,indValid);
validSetSmall=allSet(:,indValidSmall);
validClassSmall=allClass(indValidSmall);

hackit = 1;
if hackit
    trainClass = dictClass;
    trainSet = dictSet;
    trainClassSmall = dictClassSmall;
    trainSetSmall = dictSetSmall;
end

%%
save('trex_test.mat', 'dictClass', 'dictClassSmall', 'dictSet', ...
    'dictSetSmall', 'trainSet', 'trainClass', 'testSet', 'testClass',...
    'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall',...
    'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
