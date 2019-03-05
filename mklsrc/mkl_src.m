clc
clearvars -except runNum
addpath(genpath('srv1_9'));
addpath('lowlevel_functions')
load('../gen_scripts/frm_trex_pond_insitu_2class_test_insitu_single_gen_ol_0.1.mat')
% load('../gen_scripts/yale_dark.mat')
% load('../gen_scripts/yale.mat')
runNum = 1;
%%
rename=1;
if rename
    dictSetSmall   = train0data;
    dictClassSmall = train0class;
    trainSetSmall  = train0data;
    trainClassSmall = train0class;
    testSetSmall    = insitu1data;
    testClassSmall  = insitu1class;
    validSetSmall   = gen1data;
    validClassSmall = gen1class;
end
% need to sort ALL the sets
[dictClassSmall, sidxs] = sort(dictClassSmall);
dictSetSmall = dictSetSmall(:,sidxs);
[trainClassSmall, sidxs] = sort(trainClassSmall);
trainSetSmall = trainSetSmall(:,sidxs);
[testClassSmall, sidxs] = sort(testClassSmall);
testSetSmall = testSetSmall(:,sidxs);
[validClassSmall, sidxs] = sort(validClassSmall);
validSetSmall = validSetSmall(:,sidxs);

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

swapValidTest = 0;
if swapValidTest
    tempSet = testSetSmall;
    tempClass = testClassSmall;
    testSetSmall = validSetSmall;
    testClassSmall = validClassSmall;
    validSetSmall = tempSet;
    validClassSmall = tempClass;
end

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
%     p = linspace(.1,1.8,20);
%     kfncs  = { ...
%         @(x,y) x'*y; ...            % Linear
%         @(x,y) (x'*y + 1); ...
%         @(x,y) (x'*y + 0.5).^2; ...  % Polynomial
%         @(x,y) (x'*y + 0.5).^3; ...
%         @(x,y) (x'*y + 0.5).^4; ...
%         @(x,y) (x'*y + 1.0).^2; ...
%         @(x,y) (x'*y + 1.0).^3; ...
%         @(x,y) (x'*y + 1.0).^4; ...
%         @(x,y) (x'*y + 1.5).^2; ...
%         @(x,y) (x'*y + 1.5).^3; ...
%         @(x,y) (x'*y + 1.5).^4; ...
%         @(x,y) (x'*y + 2.0).^2; ...
%         @(x,y) (x'*y + 2.0).^3; ...
%         @(x,y) (x'*y + 2.0).^4; ...
%         @(x,y) (x'*y + 2.5).^2; ...
%         @(x,y) (x'*y + 2.5).^3; ...
%         @(x,y) (x'*y + 2.5).^4; ...
%         @(x,y) tanh(0.1 + 1.0*(x'*y)); ...  % Hyperbolic Tangent
%         @(x,y) tanh(0.2 + 1.0*(x'*y)); ...
%         @(x,y) tanh(0.3 + 1.0*(x'*y)); ...
%         @(x,y) tanh(0.4 + 1.0*(x'*y)); ...
%         @(x,y) tanh(0.5 + 1.0*(x'*y)); ...
%         @(x,y) tanh(0.5 + 0.2*(x'*y)); ...
%         @(x,y) tanh(0.5 + 0.4*(x'*y)); ...
%         @(x,y) tanh(0.5 + 0.6*(x'*y)); ...
%         @(x,y) tanh(0.5 + 0.8*(x'*y)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(1)^2)); ...  % Gaussian
%         @(x,y) exp((-pdist2(x.',y.').^2./p(2)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(3)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(4)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(5)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(6)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(7)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(8)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(9)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(10)^2)); ...
%                 @(x,y) exp((-pdist2(x.',y.').^2./p(11)^2)); ...  % Gaussian
%         @(x,y) exp((-pdist2(x.',y.').^2./p(12)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(13)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(14)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(15)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(16)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(17)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(18)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(19)^2)); ...
%         @(x,y) exp((-pdist2(x.',y.').^2./p(20)^2)); ...
%          };
p = linspace(.1,4,10);
kfncs  = { ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(1)^2)); ...  % Gaussian
    @(x,y) exp((-pdist2(x.',y.').^2./p(2)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(3)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(4)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(5)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(6)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(7)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(8)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(9)^2)); ...
    @(x,y) exp((-pdist2(x.',y.').^2./p(10)^2)); ...
    @(x,y) (x'*y + 1.0).^1; ...
    @(x,y) (x'*y + 1.0).^2; ...
    @(x,y) (x'*y + 1.0).^3; ...
    @(x,y) (x'*y + 0.5).^1; ...  % Polynomial
    @(x,y) (x'*y + 0.5).^2; ...
    @(x,y) (x'*y + 0.5).^3; ...
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
do_baseline = 0;
if do_baseline
    t = 0;
    h = zeros(1, length(trainClassSmall));
    htest = zeros(1, length(testClassSmall));
    hvalid = zeros(1, length(validClassSmall));
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
    errorGoal = .001;
    z = [];
    ztestcheck = [];
    zvalidcheck = [];
    while(t <= T && err>= err_thresh)
        t = t + 1;
        
        [X, h, g, z(end+1,:), zm, c, cc_rate_ini, fa_rate_ini, poor_idxs] = mklsrcUpdateWithAddition(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class,1,[], errorGoal);
        [C,CM,IND,PER] = confusion(num2bin10(trainClassSmall, length(classes)), num2bin10(h, length(classes)));
        class_percent_correct_train(:,end+1) = PER(:,3);
        
        if sum(z(t,:))/length(z(t,:)) == 1 || sum(c)==0
            err = 0;
        else
            [eta, err] = updateEta(eta, c, mu, z(t,:), zm);
        end
        
        % Add the poor samples to the dictionary
        if sum(poor_idxs) && 0 % FIXME
            for poor_idx = 1:size(poor_idxs, 1)
                if poor_idxs(poor_idx) == 1
                    Dict(:,end+1) = normc(trainSetSmall(:, poor_idx));
                    dictClassSmall(end + 1) = trainClassSmall(poor_idx);
                end
            end
            %     Rearrange Dict and add to dict classes
            [~, sort_idxs] = sort(dictClassSmall);
            dictClassSmall = dictClassSmall(sort_idxs);
            Dict = Dict(:, sort_idxs);
            dictSetSmall = Dict;
            for i=1:num_classes
                num_per_class(i) = sum(dictClassSmall == classes(i));
            end
            [Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
        end
        
        display(['TRAINING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(z(end,:))/numel(z(end,:))) ' Error: ' num2str(err)])
    end
    
    %%
    save('baseline.mat', 'z', 'ztestcheck', 'zvalidcheck', 'changes_baseline', 'changes_test', 'changes_valid', 'eta', 'validClassSmall', 'validSetSmall', 'testClassSmall', 'testSetSmall', ...
        'trainClassSmall', 'trainSetSmall', 'classes', 'h', 'htest', 'hvalid', 'num_per_class', 'Hfull', 'Gfull', 'Bfull', 'Hfulltest2', 'Gfulltest2', 'Bfulltest2', ...
        'Hfullvalid', 'Gfullvalid', 'Bfullvalid', 'kfncs', ...
        'class_percent_correct_test', 'class_percent_correct_valid', 'class_percent_correct_train',...
        'fa_rate_ini_train', 'cc_rate_ini_train', 'fa_rate_ini_test', 'cc_rate_ini_test', 'fa_rate_ini_valid', 'cc_rate_ini_valid');
else
    %% load
    load('baseline.mat')
end
%% Learn the test set now
Hfulltest = Hfull;
t = 0;
etatest = eta;
errorGoal = .001;
batchSize = floor(length(testClassSmall)/5);
numBatches = floor(length(testClassSmall)/batchSize);
available = 1:length(testClassSmall);
for smallidx = 1:numBatches
    disp(['TESTING: Generating partitioned testing matrices for partition ' num2str(smallidx) '/' num2str(numBatches) ' :  '])
    random_sample = randperm(length(available), batchSize);
    selected_samples = available(random_sample);
    available(random_sample) = []; % remove added sample from those available to add
    
    testTemp = testSetSmall(:, selected_samples);
    testTempClass = testClassSmall(selected_samples);
    
    % Need to sort the samples
    [testTempClass, sidxs] = sort(testTempClass);
    testTemp = testTemp(:,sidxs);
    
    %     testTemp = testSetSmall(:, (smallidx-1)*batchSize + 1: (smallidx-1)*batchSize + batchSize);
    %     testTempClass = testClassSmall((smallidx-1)*batchSize + 1: (smallidx-1)*batchSize + batchSize);
    
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
    
    poor_added = zeros(numel(testTempClass), 1);
    
    err = err_thresh + 1;
    t=0;
    while(t <= T && err>= err_thresh)
        t = t + 1;
        
        % This one gets the things required to update eta
        [~, ~, ~, ztest, zmtest, ctest, ~, ~, poor_idxs] = mklsrcUpdateWithAddition(Hfulltest, Gfulltest, Bfulltest, etatest, testTempClass, classes, num_per_class, 0, testTemp, errorGoal);
        
        if sum(ztest)/length(ztest) == 1
            err = 0;
        elseif sum(poor_idxs) && prod(poor_added(poor_idxs)) == 0
            err = err;
        elseif sum(ctest) == 0
            err = 0;
        else
            [etatest, err] = updateEta(etatest, ctest, mu, ztest, zmtest);
        end
        
        %             % Add the poor samples to the dictionary
        if sum(poor_idxs)
            for poor_idx = 1:size(poor_idxs, 1)
                if poor_idxs(poor_idx) == 1 && poor_added(poor_idx) == 0
                    Dict(:,end+1) = normc(testTemp(:, poor_idx));
                    dictClassSmall(end + 1) = testTempClass(poor_idx);
                end
            end
            %     Rearrange Dict and add to dict classes
            [~, sort_idxs] = sort(dictClassSmall);
            dictClassSmall = dictClassSmall(sort_idxs);
            Dict = Dict(:, sort_idxs);
            dictSetSmall = Dict;
            for i=1:num_classes
                num_per_class(i) = sum(dictClassSmall == classes(i));
            end
            [Hfulltest, Gfulltest, Bfulltest] = precomputeKernelMats(kfncs, Dict, testTemp);
            poor_added = poor_added | poor_idxs;
        end
        
        disp(['TESTING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(ztest)/numel(ztest)) ' Error: ' num2str(err)])
    end
end
% display(['TESTING: Accuracy: ' num2str(sum(ztest)/numel(ztest))])
%% 'Check OG data again'
errorGoal = .01;
disp(['TRAINING: Final check on all sets'])
[Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
[Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
[Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);

[~, htrain, ~, z(end+1,:), ~, ~, cc_rate_fin_train, fa_rate_fin_train] = mklsrcClassify(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class, 1, errorGoal);
[~, htest, ~, ztestcheck(end+1,:), ~, ~, cc_rate_fin_test, fa_rate_fin_test] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, etatest, testClassSmall, classes, num_per_class, 0, errorGoal);
[~, hvalid, ~, zvalidcheck(end+1,:), ~, ~, cc_rate_fin_valid, fa_rate_fin_valid] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, etatest, validClassSmall, classes, num_per_class, 0, errorGoal);
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

save(['results_batch' num2str(results.batchSize) '_run' num2str(runNum) '.mat'], 'results');

%% Plot?
figure(1)
[~, cm_base, ~, ~] = confusion(num2bin10(trainClassSmall, length(classes)), num2bin10(htrain, length(classes)));
imagesc(normr(cm_base));
colormap((jet))
colorbar
figure(2)
[~, cm_test, ~, ~] = confusion(num2bin10(testClassSmall, length(classes)), num2bin10(htest, length(classes)));
imagesc(normr(cm_test));
colormap((jet))
colorbar
figure(3)
[~, cm_valid, ~, ~] = confusion(num2bin10(validClassSmall, length(classes)), num2bin10(hvalid, length(classes)));
imagesc(normr(cm_valid));
colormap((jet))
colorbar