function [Train_Contacts, InSitu_Contacts1, InSitu_Contacts2, Gen_Contacts1,Gen_Contacts2] = loadAllData_jack_full(allDataPath)
%loadAllData Load the data generated from one of the gen_* scripts and
%loads it into the correct format for the ISMKL
%   Detailed explanation goes here
load(allDataPath);
shift = 1;

[train0class,trind0]=sort(train0class);
train0data=train0data(:,trind0);

[gen1class,ind1]=sort(gen1class);
[gen2class,ind2]=sort(gen2class);

[insitu1class,isind1]=sort(insitu1class);
[insitu2class,isind2]=sort(insitu2class);

gen1data=gen1data(:,ind1);
gen2data=gen2data(:,ind2);

insitu1data=insitu1data(:,isind1);
insitu2data=insitu2data(:,isind2);





% make training feature struct for classification later
for i=1:length(train0class)
    Train_Contacts(i)=struct('features', double(train0data(:,i)), 'fn','no','gt',double(train0class(i)));
    if shift
        Train_Contacts(i).gt = Train_Contacts(i).gt + 1;
    end
end

% Load Testing and Generalization Sets
for i=1:length(insitu1class)
    InSitu_Contacts1(i)=struct('features', double(insitu1data(:,i)), 'fn','no','gt',double(insitu1class(i)));
    if shift
        InSitu_Contacts1(i).gt = InSitu_Contacts1(i).gt + 1;
    end
end

% Load Testing and Generalization Sets
for i=1:length(insitu2class)
    InSitu_Contacts2(i)=struct('features', double(insitu2data(:,i)), 'fn','no','gt',double(insitu2class(i)));
    if shift
        InSitu_Contacts2(i).gt = InSitu_Contacts2(i).gt + 1;
    end
end

for i=1:length(gen1class)
    Gen_Contacts1(i)=struct('features', double(gen1data(:,i)), 'fn','no','gt',double(gen1class(i)));
    if shift
        Gen_Contacts1(i).gt = Gen_Contacts1(i).gt + 1;
    end
end

for i=1:length(gen2class)
    Gen_Contacts2(i)=struct('features', double(gen2data(:,i)), 'fn','no','gt',double(gen2class(i)));
    if shift
        Gen_Contacts2(i).gt = Gen_Contacts2(i).gt + 1;
    end
end

end

