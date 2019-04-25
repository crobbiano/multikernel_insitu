clear all
clc
% close all
%%
load('../../YALE_EXTENDED/yaleCropped.mat');
%%
% Light environments
env_light = [1:3,5:16,36:46];
% medium environments
env_medium = [17:26,47:54];
% dark environments
env_dark = [4, 27:35,55:64];
%% each illumination as a different class
illumAsClasses = 1;
classNum = 1;
if illumAsClasses
    Y_light = [];    classes_light = [];
    Y_medium = [];   classes_medium = [];
    Y_dark = [];     classes_dark = [];
    for person=1:length(people)
        if people(person).use
            for i=1:length(people(person).illumination)
                if ismember(i,env_light)
                    Y_light(:,end+1) = normc(people(person).illumination(i).img');
                    classes_light(end+1) = classNum;
                elseif ismember(i,env_medium)
                    Y_medium(:,end+1) = normc(people(person).illumination(i).img');
                    classes_medium(end+1) = classNum;
                elseif ismember(i,env_dark)
                    Y_dark(:,end+1) = normc(people(person).illumination(i).img');
                    classes_dark(end+1) = classNum;
                end
            end
            classNum = classNum + 1;
        end
    end
end

% block align the classes
[~, idx] = sort(classes_light);
classes_light = classes_light(idx);
Y_light = Y_light(:,idx);
[~, idx] = sort(classes_medium);
classes_medium = classes_medium(idx);
Y_medium = Y_medium(:,idx);
[~, idx] = sort(classes_dark);
classes_dark = classes_dark(idx);
Y_dark = Y_dark(:,idx);
%% Assign to the correct sets
trainIdx_light = randsample(length(classes_light), floor(length(classes_light)/2));
trainIdx_light = sort(trainIdx_light);
dictClass=classes_light(trainIdx_light);
dictSet=Y_light(:,trainIdx_light);
dictClassSmall=classes_light(trainIdx_light);
dictSetSmall=Y_light(:,trainIdx_light);

trainClass=classes_light(trainIdx_light);
trainSet=Y_light(:,trainIdx_light);
trainClassSmall=classes_light(trainIdx_light);
trainSetSmall=Y_light(:,trainIdx_light);

classes_nontrain = classes_dark;
Y_nontrain = Y_dark;
% classes_nontrain = classes_medium;
% Y_nontrain = Y_medium;
testIdx = randsample(length(classes_nontrain), floor(length(classes_nontrain)/2));
testIdx = sort(testIdx);
testClass=classes_nontrain(testIdx);
testSet=Y_nontrain(:,testIdx);
testClassSmall=classes_nontrain(testIdx);
testSetSmall=Y_nontrain(:,testIdx);

validIdx_light = setdiff(1:length(classes_light),trainIdx_light);
validIdx = setdiff(1:length(classes_nontrain),testIdx);
validClass=[classes_light(validIdx_light) classes_nontrain(validIdx)];
validSet=[Y_light(:,validIdx_light) Y_nontrain(:,validIdx)];
validClassSmall=[classes_light(validIdx_light) classes_nontrain(validIdx) classes_medium];
validSetSmall=[Y_light(:,validIdx_light) Y_nontrain(:,validIdx) Y_medium];
[~, idx] = sort(validClass);
validClass = validClass(idx);
validSet = validSet(:,idx);
[~, idx] = sort(validClassSmall);
validClassSmall = validClassSmall(idx);
validSetSmall = validSetSmall(:,idx);

%%
save('yale_darkIS_darkMediumGen.mat', 'dictClass', 'dictClassSmall', 'dictSet', ...
    'dictSetSmall', 'trainSet', 'trainClass', 'testSet', 'testClass',...
    'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall',...
    'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
