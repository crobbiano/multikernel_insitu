function [Train_Contacts, Test_Contacts, Gen_Contacts] = loadAllData(allDataPath)
%loadAllData Load the data generated from one of the gen_* scripts and
%loads it into the correct format for the ISMKL
%   Detailed explanation goes here
load(allDataPath);
shift = 1;
% make training feature struct for classification later
for i=1:length(dictClassSmall)
    Train_Contacts(i)=struct('features', double(dictSetSmall(:,i)), 'fn','no','gt',double(dictClassSmall(i)));
    if shift
        Train_Contacts(i).gt = Train_Contacts(i).gt + 1;
    end
end

% Load Testing and Generalization Sets
for i=1:length(testClassSmall)
    Test_Contacts(i)=struct('features', double(testSetSmall(:,i)), 'fn','no','gt',double(testClassSmall(i)));
    if shift
        Test_Contacts(i).gt = Test_Contacts(i).gt + 1;
    end
end

for i=1:length(validClassSmall)
    Gen_Contacts(i)=struct('features', double(validSetSmall(:,i)), 'fn','no','gt',double(validClassSmall(i)));
    if shift
        Gen_Contacts(i).gt = Gen_Contacts(i).gt + 1;
    end
end
end

