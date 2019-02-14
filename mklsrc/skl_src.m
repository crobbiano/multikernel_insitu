%% Single kernel version
% Here, we will search for the kernel that provides the best classification
% abilities on the training set, then apply that kernel only.
clc
clearvars -except runNum
addpath(genpath('srv1_9'));
addpath('lowlevel_functions')
%load('../gen_scripts/trex_insitu_2class_test_insitu.mat')
load('../datasets/frm_trex_pond_insitu_2class_test_insitu_single_gen_ol_0.1.mat');
% load('../gen_scripts/yale.mat')
%%
rename=1;
if rename
    dictSetSmall   = train0data;
    dictClassSmall = train0class;
    trainSetSmall = train0data;
    trainClassSmall = train0class;
    testSetSmall    = insitu1data;
    testClassSmall  = insitu1class;
    validSetSmall   = gen1data;
    validClassSmall = gen1class;    
end

% need to sort the dict set
[dictClassSmall, sidxs] = sort(dictClassSmall);
dictSetSmall = dictSetSmall(:,sidxs);

% More hacking
hack = 0;
if hack
    set_temp = trainSetSmall;
    class_temp = trainClassSmall;
    trainSetSmall = testSetSmall;
    trainClassSmall = testClassSmall;
    testSetSmall = validSetSmall;
    testClassSmall = validClassSmall;
    validSetSmall = set_temp;
    validClassSmall = class_temp;
    clear set_temp class_temp
end

dictSetSmall = normc(dictSetSmall);
trainSetSmall = normc(trainSetSmall);
testSetSmall = normc(testSetSmall);
validSetSmall = normc(validSetSmall);

shiftClasses = 1;
if shiftClasses
    dictClassSmall = dictClassSmall + 1;
    trainClassSmall = trainClassSmall + 1;
    testClassSmall = testClassSmall + 1;
    validClassSmall = validClassSmall + 1;
end

% Save the dictionary
Dict = dictSetSmall;

% X should be N by M where N is number of atoms and M is number of testing samples.
X = zeros(size(Dict, 2), size(trainSetSmall, 2));
%% Generate kernel fncs
kfncs  = { ...
    @(x,y) x'*y; ...            % Linear
    @(x,y) (x'*y + 1); ...
    @(x,y) (x'*y + 0.5).^2; ...  % Polynomial
    @(x,y) (x'*y + 0.5).^3; ...
    @(x,y) (x'*y + 0.5).^4; ...
    @(x,y) (x'*y + 1.0).^2; ...
    @(x,y) (x'*y + 1.0).^3; ...
    @(x,y) (x'*y + 1.0).^4; ...
    @(x,y) (x'*y + 1.5).^2; ...
    @(x,y) (x'*y + 1.5).^3; ...
    @(x,y) (x'*y + 1.5).^4; ...
    @(x,y) (x'*y + 2.0).^2; ...
    @(x,y) (x'*y + 2.0).^3; ...
    @(x,y) (x'*y + 2.0).^4; ...
    @(x,y) (x'*y + 2.5).^2; ...
    @(x,y) (x'*y + 2.5).^3; ...
    @(x,y) (x'*y + 2.5).^4; ...
    @(x,y) tanh(0.1 + 1.0*(x'*y)); ...  % Hyperbolic Tangent
    @(x,y) tanh(0.2 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.3 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.4 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.2*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.4*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.6*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.8*(x'*y)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.1)); ...  % Gaussian
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.8)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.9)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.0)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.1)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.8)); ...
    };
%% Compute the M kernel matrices
display(['BASELINE: Generating first set of matrices: '])
% Find K_m(Y, Y) for all M kernel functions
kernel_mats = cell(length(kfncs), 1);
parfor_progress(length(kfncs))
for m=1:length(kfncs)
    option.kernel = 'cust'; option.kernelfnc=kfncs{m};
    kernel_mats{m} = computeKernelMatrix(Dict,Dict,option);
end
parfor_progress(0)

% Make the ideal matrix - assumes blocking of dictionary and data
K_ideal = eye(size(Dict,2));
% Find the number of samples in each class
classes = unique(dictClassSmall);
num_classes = numel(classes);
masks = zeros(size(Dict,2),numel(classes));
num_samples_per_class = zeros(num_classes, 1);
for i=1:num_classes
    num_samples_per_class(i) = sum(dictClassSmall == classes(i));
    masks(:,i) = dictClassSmall == classes(i);
    locs = find(dictClassSmall == classes(i));
    K_ideal(min(locs):max(locs),min(locs):max(locs)) = 1;
end
%% Get ranked ordering of kfncs based on similarity to ideal kernel
display(['BASELINE: Generating ranks of matrices: '])
parfor_progress(length(kfncs));
alignment_scores = zeros(length(kfncs),1);
for i=1:length(kfncs)
    alignment_scores(i) = kernelAlignment(kernel_mats{i}, K_ideal);
    parfor_progress();
end
parfor_progress(0);
[sorted, idx] = sort(alignment_scores,'descend');
kernel_mats = kernel_mats(idx);
kfncs = kfncs(idx);
%% Compute more kernel matrices
[Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
[Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
[Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);
%% Parameters
mu = .5;
% sparsity_reg \lambda
lambda = 1;
% max iterations
T = 10;
% error thresh for convergence
err_thresh = .01;
err = err_thresh + 1;

% Make eta
eta = zeros(length(kfncs),1);
eta(1)=1;

% Find the number of samples in each class
classes = unique(dictClassSmall);
num_classes = numel(classes);
num_per_class = zeros(num_classes,1);
for i=1:num_classes
    num_per_class(i) = sum(dictClassSmall == classes(i));
end
%% Loop to get all sparse coeffs
do_baseline = 1;
if do_baseline
    class_percent_correct_train = [];
    class_percent_correct_test = [];
    class_percent_correct_valid = [];
    changes_baseline = [];
    changes_test = [];
    changes_valid = [];
    fa_rate_ini_train = [];
    cc_rate_ini_train = [];
    fa_rate_ini_test = [];
    cc_rate_ini_test = [];
    fa_rate_ini_valid = [];
    cc_rate_ini_valid = [];
    z = [];
    ztestcheck = [];
    zvalidcheck = [];
    errGoal=0.1;
    
    [X, h, g, z(end+1,:), zm, c, cc_rate_ini, fa_rate_ini, poor_idxs] = mklsrcUpdateWithAddition(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class,1, [], errGoal);
    [C,CM,IND,PER] = confusion(num2bin10(trainClassSmall, length(classes)), num2bin10(h, length(classes)));
    class_percent_correct_train(:,end+1) = PER(:,3);
    
    % find the kernel that performed the best and choose that as our only kernel
    [~, bestKernelIdx] = max(sum(zm, 2));
    bestKernelIdx = bestKernelIdx(1); % pick just the first if there are many
    eta = zeros(length(kfncs),1);
    eta(bestKernelIdx)=1;
    
    display(['TRAINING: Accuracy: ' num2str(sum(z(end,:))/numel(z(end,:)))])
    %%
    save('baseline_single.mat', 'z', 'ztestcheck', 'zvalidcheck', 'changes_baseline', 'changes_test', 'changes_valid', 'eta', 'validClassSmall', 'validSetSmall', 'testClassSmall', 'testSetSmall', ...
        'trainClassSmall', 'trainSetSmall', 'classes', 'h', 'num_per_class', 'Hfull', 'Gfull', 'Bfull', 'Hfulltest2', 'Gfulltest2', 'Bfulltest2', ...
        'Hfullvalid', 'Gfullvalid', 'Bfullvalid', 'kfncs', ...
        'class_percent_correct_test', 'class_percent_correct_valid', 'class_percent_correct_train',...
        'fa_rate_ini_train', 'cc_rate_ini_train', 'fa_rate_ini_test', 'cc_rate_ini_test', 'fa_rate_ini_valid', 'cc_rate_ini_valid');
else
    %% load
    load('baseline_single.mat')
end
%% Learn the test set now
Hfulltest = Hfull;
etatest = eta;
batchSize = length(testClassSmall); % only use one batch, no updating
numBatches = floor(length(testClassSmall)/batchSize);
available = 1:length(testClassSmall);
for smallidx = 1:numBatches
    disp(['TESTING: Generating partitioned testing matrices for partition ' num2str(smallidx) '/' num2str(numBatches) ' :  '])
    random_sample = randperm(length(available), batchSize);
    selected_samples = available(random_sample);
    available(random_sample) = []; % remove added sample from those available to add
    
    testTemp = testSetSmall(:, selected_samples);
    testTempClass = testClassSmall(selected_samples);
    
    Gfulltest = cell(length(kfncs), size(testTemp, 2));
    Bfulltest = cell(length(kfncs), size(testTemp, 2));
    for kidx=1:length(kfncs)
        eta_temp = zeros(length(kfncs),1);
        eta_temp(kidx) = 1; % place a 1 in the current kernel
        
        for sidx=1:size(testTemp, 2)
            Gfulltest{kidx,sidx}=computeMultiKernelMatrix(Dict,testTemp(:,sidx),eta_temp,kfncs);
            Bfulltest{kidx,sidx}=computeMultiKernelMatrix(testTemp(:,sidx),testTemp(:,sidx),eta_temp,kfncs);
        end
    end
          
    % This one gets the things required to update eta
    [~, ~, ~, ztest, zmtest, ctest, ~, ~, poor_idxs] = mklsrcUpdateWithAddition(Hfulltest, Gfulltest, Bfulltest, etatest, testTempClass, classes, num_per_class, 0, testTemp, errGoal);

    disp(['TESTING: Accuracy: ' num2str(sum(ztest)/numel(ztest))])
end
%% 'Check OG data again'
disp(['TRAINING: Final check on all sets'])
[Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
[Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
[Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);

[~, htrain, ~, z(end+1,:), ~, ~, cc_rate_fin_train, fa_rate_fin_train] = mklsrcClassify(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class, 0, errGoal);
[~, htest, ~, ztestcheck(end+1,:), ~, ~, cc_rate_fin_test, fa_rate_fin_test] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, etatest, testClassSmall, classes, num_per_class, 0, errGoal);
[~, hvalid, ~, zvalidcheck(end+1,:), ~, ~, cc_rate_fin_valid, fa_rate_fin_valid] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, etatest, validClassSmall, classes, num_per_class, 0, errGoal);
validacc2 = sum(z(end,:))/numel(z(end,:));
display(['TRAINING: Post-Accuracy: ' num2str(sum(z(end,:))/numel(z(end,:)))])
display(['TESTING: Post-Accuracy: ' num2str(sum(ztestcheck(end,:))/numel(ztestcheck(end,:)))])
display(['GENERAL: Post-Accuracy: ' num2str(sum(zvalidcheck(end,:))/numel(zvalidcheck(end,:)))])
%% save things
results.class_percent_correct_train = class_percent_correct_train;
results.class_percent_correct_test = class_percent_correct_test;
results.class_percent_correct_valid = class_percent_correct_valid;
results.fa_rate_fin_train = fa_rate_fin_train;
results.cc_rate_fin_train = cc_rate_fin_train;
results.fa_rate_fin_test = fa_rate_fin_test;
results.cc_rate_fin_test = cc_rate_fin_test;
results.fa_rate_fin_valid = fa_rate_fin_valid;
results.cc_rate_fin_valid = cc_rate_fin_valid;
results.fa_rate_ini_train = fa_rate_ini_train;
results.cc_rate_ini_train = cc_rate_ini_train;
results.fa_rate_ini_test = fa_rate_ini_test;
results.cc_rate_ini_test = cc_rate_ini_test;
results.fa_rate_ini_valid = fa_rate_ini_valid;
results.cc_rate_ini_valid = cc_rate_ini_valid;
results.changes_baseline = changes_baseline;
results.changes_test = changes_test;
results.changes_valid = changes_valid;
results.eta = eta;
results.etatest = etatest;
results.z = z;
results.ztest = ztest;
results.batchSize = batchSize;

save(['results_single_batch' num2str(results.batchSize) '.mat'], 'results');
%% Plot?
figure(1)
plotconfusion(num2bin10(trainClassSmall, length(classes)), num2bin10(htrain, length(classes)));
figure(2)
plotconfusion(num2bin10(testClassSmall, length(classes)), num2bin10(htest, length(classes)));
figure(3)
plotconfusion(num2bin10(validClassSmall, length(classes)), num2bin10(hvalid, length(classes)));
