clear all
clc
close all
%%
load('../../YALE_EXTENDED/yaleCropped.mat');

%% each person as a different class
peopleAsClasses = 0;
if peopleAsClasses
    numClasses = length(people);
    % use poses 005, 010, and 015 for baseline
    Y_baseline = [];
    classes_baseline = [];
    Y_insitu = [];
    classes_insitu = [];
    Y_gen = [];
    classes_gen = [];
    for person=1:length(people)
        for j=1:length(people(person).illumination)
            if contains(people(person).illumination(j).condition, {'005','010','015'})
                Y_baseline(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_baseline(end+1) = person;
            elseif contains(people(person).illumination(j).condition, {'050','060','070'})
                Y_insitu(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_insitu(end+1) = person;
            elseif contains(people(person).illumination(j).condition, {'110'})
                Y_gen(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_gen(end+1) = person;
            end
        end
    end
end
%% each illumination as a different class
peopleAsClasses = 0;
if peopleAsClasses
    numClasses = length(people);
    % use poses 005, 010, and 015 for baseline
    Y_baseline = [];
    classes_baseline = [];
    Y_insitu = [];
    classes_insitu = [];
    Y_gen = [];
    classes_gen = [];
    for person=1:length(people)
        for j=1:length(people(person).illumination)
            if contains(people(person).illumination(j).condition, {'005','010','015'})
                Y_baseline(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_baseline(end+1) = person;
            elseif contains(people(person).illumination(j).condition, {'050','060','070'})
                Y_insitu(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_insitu(end+1) = person;
            elseif contains(people(person).illumination(j).condition, {'110'})
                Y_gen(:,end+1) = normc(reshape(imresize(people(person).illumination(j).img,.2),[],1));
                classes_gen(end+1) = person;
            end
        end
    end
end
%% Assign to the correct sets
dictClass=classes_baseline;
dictSet=Y_baseline;
dictClassSmall=classes_baseline;
dictSetSmall=Y_baseline;

trainClass=classes_baseline;
trainSet=Y_baseline;
trainClassSmall=classes_baseline;
trainSetSmall=Y_baseline;

testClass=classes_insitu;
testSet=Y_insitu;
testClassSmall=classes_insitu;
testSetSmall=Y_insitu;

validClass=classes_gen;
validSet=Y_gen;
validClassSmall=classes_gen;
validSetSmall=Y_gen;
%%
save('yale.mat', 'dictClass', 'dictClassSmall', 'dictSet', ...
    'dictSetSmall', 'trainSet', 'trainClass', 'testSet', 'testClass',...
    'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall',...
    'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
