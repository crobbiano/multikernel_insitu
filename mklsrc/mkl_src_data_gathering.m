for runNum=1:1
    clc
    clearvars -except runNum
    addpath(genpath('srv1_9'));
    addpath('lowlevel_functions')
    load('../gen_scripts/yale_dark.mat')
%     load('../gen_scripts/frm_trex_pond_insitu_2class_test_insitu_single_gen_ol_0.1.mat')
    %%
    rename=0;
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
    
    shiftClasses = 0;
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
    textprogressbar('BASELINE: Generating first set of matrices: ')
    % Find K_m(Y, Y) for all M kernel functions
    kernel_mats = cell(length(kfncs), 1);
    for m=1:length(kfncs)
        option.kernel = 'cust'; option.kernelfnc=kfncs{m};
        kernel_mats{m} = computeKernelMatrix(Dict,Dict,option);
        textprogressbar(m*100/length(kfncs));
    end
    
    % Make the ideal matrix - FIXME - assumes blocks of samples (probably fine)
    K_ideal = eye(size(Dict,2));
    % Find the number of samples in each class
    classes = unique(dictClassSmall);
    num_classes = numel(classes);
    masks = zeros(size(Dict,2),numel(classes));
    for i=1:num_classes
        num_samples_per_class(i) = sum(dictClassSmall == classes(i));
        masks(:,i) = dictClassSmall == classes(i);
        locs = find(dictClassSmall == classes(i));
        K_ideal(min(locs):max(locs),min(locs):max(locs)) = 1;
    end
    textprogressbar(' ');
    %% Get ranked ordering of kfncs based on similarity to ideal kernel
    textprogressbar('BASELINE: Generating ranks of matrices: ')
    for i=1:length(kfncs)
        alignment_scores(i) = kernelAlignment(kernel_mats{i}, K_ideal);
        textprogressbar(i*100/length(kfncs));
    end
    [sorted, idx] = sort(alignment_scores,'descend');
    kernel_mats = kernel_mats(idx);
    kfncs = kfncs(idx);
    textprogressbar(' ');
    %% Compute more kernel matrices
    [Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
    [Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
    [Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);
    %% Parameters
    mu = .05;
    % sparsity_reg \lambda
    lambda = 1;
    % max iterations
    T = 10;
    % error thresh for convergence
    err_thresh = .01;
    err = err_thresh + 1;
    
    results.atoms_added = 0;
    
    % Make eta
    eta = zeros(length(kfncs),1);
    eta(1)=1;
    
    % Find the number of samples in each class
    classes = unique(dictClassSmall);
    num_classes = numel(classes);
    for i=1:num_classes
        num_per_class(i) = sum(dictClassSmall == classes(i));
    end
    %% Loop to get all sparse coeffs
    errorGoal = .001;
    do_baseline = 1;
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
        z = [];
        ztestcheck = [];
        zvalidcheck = [];
        while(t <= T && err>= err_thresh)
            t = t + 1;
            
            hprev = h;
            hprevtest = htest;
            hprevvalid = hvalid;
            
            [X, h, g, z(end+1,:), zm, c, cc_rate_ini, fa_rate_ini, poor_idxs] = mklsrcUpdateWithAddition(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class,1, [], errorGoal);
            [C,CM,IND,PER] = confusion(num2bin10(trainClassSmall, length(classes)), num2bin10(h, length(classes)));
            class_percent_correct_train(:,end+1) = PER(:,3);
            
            
            [~, htest, ~, ztestcheck(end+1, :), ~, ~] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, eta, testClassSmall, classes, num_per_class, 0, errorGoal);
            [C,CM,IND,PER] = confusion(num2bin10(testClassSmall, max(classes)), num2bin10(htest, max(classes)));
            class_percent_correct_test(:,end+1) = PER(:,3);
            
            [~, hvalid, ~, zvalidcheck(end+1, :), ~, ~] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, eta, validClassSmall, classes, num_per_class, 0, errorGoal);
            [C,CM,IND,PER] = confusion(num2bin10(validClassSmall, max(classes)), num2bin10(hvalid, max(classes)));
            class_percent_correct_valid(:,end+1) = PER(:,3);
            
            %             Compare the hamming distances
            if t > 1
                changes_baseline(end+1,:) = hprev ~= h;
                changes_test(end+1,:) = hprevtest ~= htest;
                changes_valid(end+1,:) = hprevvalid ~= hvalid;
            end
            
            if sum(z(t,:))/length(z(t,:)) == 1 | sum(c)==0
                err = 0;
            else
                [eta, err] = updateEta(eta, c, mu, z(t,:), zm);
            end
            
            % Add the poor samples to the dictionary
            if sum(poor_idxs) & 0 % FIXME
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
        
        %     [~, ~, ~, zcheck, ~, ~, cc_rate_ini_train, fa_rate_ini_train] = mklsrcClassify(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class, 1);
        %     [~, ~, ~, ztestcheck, ~, ~, cc_rate_ini_test, fa_rate_ini_test] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, eta, testClassSmall, classes, num_per_class, 0);
        %     [~, ~, ~, zvalidcheck, ~, ~, cc_rate_ini_valid, fa_rate_ini_valid] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, eta, validClassSmall, classes, num_per_class, 0);
        %%
        save('baseline.mat', 'z', 'ztestcheck', 'zvalidcheck', 'changes_baseline', 'changes_test', 'changes_valid', 'eta', 'validClassSmall', 'validSetSmall', 'testClassSmall', 'testSetSmall', ...
            'trainClassSmall', 'trainSetSmall', 'classes', 'h', 'htest', 'hprev', 'hvalid',  'hprevtest', 'hprevvalid', 'num_per_class', 'Hfull', 'Gfull', 'Bfull', 'Hfulltest2', 'Gfulltest2', 'Bfulltest2', ...
            'Hfullvalid', 'Gfullvalid', 'Bfullvalid', 'kfncs', ...
            'class_percent_correct_test', 'class_percent_correct_valid', 'class_percent_correct_train',...
            'fa_rate_ini_train', 'cc_rate_ini_train', 'fa_rate_ini_test', 'cc_rate_ini_test', 'fa_rate_ini_valid', 'cc_rate_ini_valid')
        
%         [Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
%         [Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);
        [~, ~, ~, zpre, ~, ~, cc_rate_fin_train, fa_rate_fin_train] = mklsrcClassify(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class, 1, errorGoal);
        [~, ~, ~, ztestcheckpre, ~, ~, cc_rate_fin_test, fa_rate_fin_test] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, eta, testClassSmall, classes, num_per_class, 0, errorGoal);
        [~, ~, ~, zvalidcheckpre, ~, ~, cc_rate_fin_valid, fa_rate_fin_valid] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, eta, validClassSmall, classes, num_per_class, 0, errorGoal);
        
        validacc2 = sum(z(end,:))/numel(z(end,:));
        display(['TRAINING: Pre-Accuracy: ' num2str(sum(zpre)/numel(zpre))])
        display(['TESTING: Pre-Accuracy: ' num2str(sum(ztestcheckpre)/numel(ztestcheckpre))])
        display(['GENERAL: Pre-Accuracy: ' num2str(sum(zvalidcheckpre)/numel(zvalidcheckpre))])
    else
        %% load
        load('baseline.mat')
    end
    %% Learn the test set now
    Hfulltest = Hfull;
    t = 0;
    etatest = eta;
        batchSize = floor(length(testClassSmall)/2);
%     batchSize = 25;
    numBatches = floor(length(testClassSmall)/batchSize);
    %     numBatches = floor(500/batchSize);
    % numBatches = 20;
    % numBatches = 1000/batchSize;
    available = 1:length(testClassSmall);
    for smallidx = 1:numBatches
        textprogressbar(['TESTING: Generating partitioned testing matrices for partition ' num2str(smallidx) ' :  '])
        random_sample = randperm(length(available), batchSize);
        selected_samples = available(random_sample);
        available(random_sample) = []; % remove added sample from those available to add
        %     new_in_situ_samp((smallidx-1)*batchSize + 1:(smallidx-1)*batchSize + batchSize) = selected_samples;
        
        testTemp = testSetSmall(:, selected_samples);
        testTempClass = testClassSmall(selected_samples);
        
        % Need to sort the samples
        [testTempClass, sidxs] = sort(testTempClass);
        testTemp = testTemp(:,sidxs);
        
        %         testTemp = testSetSmall(:, (smallidx-1)*batchSize + 1: (smallidx-1)*batchSize + batchSize);
        %         testTempClass = testClassSmall((smallidx-1)*batchSize + 1: (smallidx-1)*batchSize + batchSize);
        
        %         Hfulltest = Hfull;
        
        Gfulltest = cell(length(kfncs), size(testTemp, 2));
        Bfulltest = cell(length(kfncs), size(testTemp, 2));
        for kidx=1:length(kfncs)
            eta_temp = [];
            eta_temp(kidx) = 1; % place a 1 in the current kernel
            
            for sidx=1:size(testTemp, 2)
                Gfulltest{kidx,sidx}=computeMultiKernelMatrix(Dict,testTemp(:,sidx),eta_temp,kfncs);
                Bfulltest{kidx,sidx}=computeMultiKernelMatrix(testTemp(:,sidx),testTemp(:,sidx),eta_temp,kfncs);
            end
            textprogressbar(kidx*100/length(kfncs));
        end
        %         %
        %         [Hfulltest, Gfulltest, Bfulltest] = precomputeKernelMats(kfncs, Dict, testTemp);
        textprogressbar(' ');
        
        poor_added = zeros(numel(testTempClass), 1);
        
        err = err_thresh + 1;
        t=0;
        while(t <= T && err>= err_thresh)
            t = t + 1;
            
            % This one gets the things required to update eta
            %         [~, ~, ~, ztest, zmtest, ctest] = mklsrcUpdate(Hfulltest, Gfulltest, Bfulltest, etatest, testTempClass, classes, num_per_class);
            [~, ~, ~, ztest, zmtest, ctest, ~, ~, poor_idxs] = mklsrcUpdateWithAddition(Hfulltest, Gfulltest, Bfulltest, etatest, testTempClass, classes, num_per_class, 0, testTemp, errorGoal);
            
            if sum(ztest)/length(ztest) == 1 
                err = 0;
            elseif sum(ctest)==0
%                 err = 0;
                err = err;
            else
                [etatest, err] = updateEta(etatest, ctest, mu, ztest, zmtest);
            end
            
            %             % Add the poor samples to the dictionary
            if sum(poor_idxs)
                for poor_idx = 1:size(poor_idxs, 1)
                    if poor_idxs(poor_idx) == 1 && poor_added(poor_idx) == 0
                        Dict(:,end+1) = normc(testSetSmall(:, poor_idx));
                        dictClassSmall(end + 1) = testClassSmall(poor_idx);
                        results.atoms_added = results.atoms_added + 1;
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
            
            display(['TESTING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(ztest)/numel(ztest)) ' Error: ' num2str(err)])
            %         display(['TESTING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(ztestcheck(end, :))/numel(ztestcheck(end, :))) ' Error: ' num2str(err)])
        end
        
        % Add the poor samples to the dictionary
        if sum(poor_idxs)
            for poor_idx = 1:size(poor_idxs, 1)
                if poor_idxs(poor_idx) == 1 && poor_added(poor_idx) == 0
                    Dict(:,end+1) = normc(testSetSmall(:, poor_idx));
                    dictClassSmall(end + 1) = testClassSmall(poor_idx);
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
        
        hprev = h;
        hprevtest = htest;
        hprevvalid = hvalid;
        
        [Hfull, Gfull, Bfull] = precomputeKernelMats(kfncs, Dict, trainSetSmall);
        [Hfulltest2, Gfulltest2, Bfulltest2] = precomputeKernelMats(kfncs, Dict, testSetSmall);
        [Hfullvalid, Gfullvalid, Bfullvalid] = precomputeKernelMats(kfncs, Dict, validSetSmall);
        
        % This one gets the things to calculate classification on all testing samples
        [~, htest, ~, ztestcheck(end+1, :), ~, ~, ~, ~] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, etatest, testClassSmall, classes, num_per_class, 0, errorGoal);
        [~,~,~,PER] = confusion(num2bin10(testClassSmall, max(classes)), num2bin10(htest, max(classes)));
        class_percent_correct_test(:,end+1) = PER(:,3);
        
        % generalization set
        [~, hvalid, ~, zvalidcheck(end+1, :), ~, ~, ~, ~] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, eta, validClassSmall, classes, num_per_class, 0, errorGoal);
        [~,~,~,PER] = confusion(num2bin10(validClassSmall, max(classes)), num2bin10(hvalid, max(classes)));
        class_percent_correct_valid(:,end+1) = PER(:,3);
        
        % more baseline stuff
        [~, h, ~, z(end+1,:), zm, c, ~, ~] = mklsrcClassify(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class, 1, errorGoal);
        [~,~,~,PER] = confusion(num2bin10(trainClassSmall, max(classes)), num2bin10(h, max(classes)));
        class_percent_correct_train(:,end+1) = PER(:,3);
        
        % Compare the hamming distances
        changes_baseline(end+1,:) = hprev ~= h;
        changes_test(end+1,:) = hprevtest ~= htest;
        changes_valid(end+1,:) = hprevvalid ~= hvalid;
    end
    % display(['TESTING: Accuracy: ' num2str(sum(ztest)/numel(ztest))])
    %% 'Check OG data again'
    display(['TRAINING: 2nd check'])
    
    [~, ~, ~, z(end+1,:), ~, ~, cc_rate_fin_train, fa_rate_fin_train] = mklsrcClassify(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class, 1, errorGoal);
    [~, ~, ~, ztestcheck(end+1,:), ~, ~, cc_rate_fin_test, fa_rate_fin_test] = mklsrcClassify(Hfulltest2, Gfulltest2, Bfulltest2, etatest, testClassSmall, classes, num_per_class, 0, errorGoal);
    [~, ~, ~, zvalidcheck(end+1,:), ~, ~, cc_rate_fin_valid, fa_rate_fin_valid] = mklsrcClassify(Hfullvalid, Gfullvalid, Bfullvalid, etatest, validClassSmall, classes, num_per_class, 0, errorGoal);
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
    
    save(['results/results_batch' num2str(results.batchSize) '_run' num2str(runNum) '.mat'], 'results');
end
%% Make some figures
oldPlotting = 0;
if oldPlotting
    % More baseline
    figure(96); clf;
    idx = 1;
    clear cc b bb
    hold on
    text_vec = 1:2:size(class_percent_correct_train,2);
    for i1=1:size(class_percent_correct_train, 1)
        if sum(class_percent_correct_train(i1, :)) > 0
            color = [mod((3*i1/15),1) mod((7*(i1+1)/15),1) mod(1-(i1/10),1)];
            subplot(11,3,i1)
            b(idx) = plot(class_percent_correct_train(i1, :),'--', 'Color', color);
            text(text_vec, class_percent_correct_train(i1, text_vec), num2str(i1-1))
            cc{idx} = [num2str(i1-1) ' - Test'];
            idx = idx + 1;
            grid minor
            %         ylim([.01 1])
            ylabel([num2str(i1-1) ' - Correct Classification Rate'])
            xlabel('In-Situ Learning Iteration');
            ylim([.5 1])
            legend('Baseline')
        end
    end
    
    figure(989);clf
    idx = 1;
    clear cc b bb
    hold on
    text_vec = 1:2:size(class_percent_correct_test,2);
    for i2=1:size(class_percent_correct_test, 1)
        if sum(class_percent_correct_test(i2, :)) > 0
            color = [mod((3*i2/15),1) mod((7*(i2+1)/15),1) mod(1-(i2/10),1)];
            subplot(11,3,i2)
            b(idx) = plot(class_percent_correct_test(i2, :),'--', 'Color', color);
            text(text_vec, class_percent_correct_test(i2, text_vec), num2str(i2-1))
            cc{idx} = [num2str(i2-1) ' - Test'];
            idx = idx + 1;
            grid minor
            %         ylim([.01 1])
            ylabel([num2str(i2-1) ' - Correct Classification Rate'])
            xlabel('In-Situ Learning Iteration');
            ylim([.5 1])
            legend('In-Situ')
        end
    end
    
    % idx = 1;
    % clear cc b bb
    % hold on
    % text_vec = 1:2:size(class_percent_correct_valid,2);
    % for i=1:size(class_percent_correct_valid, 1)
    %     if sum(class_percent_correct_valid(i, :)) > 0
    %         color = [mod((3*i/15),1) mod((7*(i+1)/15),1) 1-(i/10)];
    %         subplot(3,2,i2+i1+i)
    %         b(idx) = plot(class_percent_correct_valid(i, :),'--', 'Color', color);
    %         text(text_vec, class_percent_correct_valid(i, text_vec), num2str(i-1))
    %         cc{idx} = [num2str(i-1) ' - Test'];
    %         idx = idx + 1;
    %         grid minor
    %         %         ylim([.01 1])
    %         ylabel([num2str(i-1) ' - Correct Classification Rate'])
    %         xlabel('In-Situ Learning Iteration');
    %         ylim([.5 1])
    %         legend('Generalization')
    %     end
    % end
    
    %%%%%%%%%%%%%%%%%%%%
    figure(919); clf; hold on
    subplot(3,1,1)
    hold on
    clear bb cc
    text_vec = 1:500:size(fa_rate_fin_train, 2);
    for i=1:size(cc_rate_fin_train,1)
        color = [mod((3*i/15),1) mod((6*(i+1)/15),1) mod(1-(i/10),1)];
        bb(i) = plot(fa_rate_fin_train(i,:), cc_rate_fin_train(i,:),'--', 'Color', color);
        %     text(results.fa_rate_nonan(i, text_vec), results.cc_rate_nonan(i, text_vec), num2str(i-1));
        cc{i} = [num2str(i-1) ''];
    end
    % ylim([.75 1]);
    grid on;
    legend(cc)
    ylabel('Probability of correct classification')
    xlabel('Probability of false alarm')
    legend('Baseline - 0', '1')
    
    subplot(3,1,2)
    hold on
    clear bb cc
    text_vec = 1:500:size(fa_rate_fin_test, 2);
    for i=1:size(cc_rate_fin_test,1)
        color = [mod((3*i/15),1) mod((6*(i+1)/15),1) mod(1-(i/10),1)];
        bb(i) = plot(fa_rate_fin_test(i,:), cc_rate_fin_test(i,:),'--', 'Color', color);
        %     text(results.fa_rate_nonan(i, text_vec), results.cc_rate_nonan(i, text_vec), num2str(i-1));
        cc{i} = [num2str(i-1) ''];
    end
    % ylim([.75 1]);
    grid on;
    legend(cc)
    ylabel('Probability of correct classification')
    xlabel('Probability of false alarm')
    legend('In-Situ - 0', '1')
    
    subplot(3,1,3)
    hold on
    clear bb cc
    text_vec = 1:500:size(fa_rate_fin_valid, 2);
    for i=1:size(cc_rate_fin_valid,1)
        color = [mod((3*i/15),1) mod((6*(i+1)/15),1) mod(1-(i/10),1)];
        bb(i) = plot(fa_rate_fin_valid(i,:), cc_rate_fin_valid(i,:),'--', 'Color', color);
        %     text(results.fa_rate_nonan(i, text_vec), results.cc_rate_nonan(i, text_vec), num2str(i-1));
        cc{i} = [num2str(i-1) ''];
    end
    % ylim([.75 1]);
    grid on;
    legend(cc)
    ylabel('Probability of correct classification')
    xlabel('Probability of false alarm')
    legend('Generalization - 0', '1')
    %%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(906); clf;
    subplot(3, 1, 1);hold on
    fa_rate_fin_avg_train = nanmean(fa_rate_fin_train);
    cc_rate_fin_avg_train = nanmean(cc_rate_fin_train);
    fa_rate_ini_avg_train = nanmean(fa_rate_ini_train);
    cc_rate_ini_avg_train = nanmean(cc_rate_ini_train);
    plot(fa_rate_ini_avg_train,cc_rate_ini_avg_train,'LineWidth',2);
    plot(fa_rate_fin_avg_train,cc_rate_fin_avg_train,'r','LineWidth',2)
    xlabel('P_{FA}')
    ylabel('P_{CC}')
    legend('Baseline: After Baseline Training','Baseline: After In-Situ Learning')
    grid on
    
    subplot(3, 1, 2);hold on
    fa_rate_fin_avg_test = nanmean(fa_rate_fin_test);
    cc_rate_fin_avg_test = nanmean(cc_rate_fin_test);
    fa_rate_ini_avg_test = nanmean(fa_rate_ini_test);
    cc_rate_ini_avg_test = nanmean(cc_rate_ini_test);
    plot(fa_rate_ini_avg_test,cc_rate_ini_avg_test,'LineWidth',2);
    plot(fa_rate_fin_avg_test,cc_rate_fin_avg_test,'r','LineWidth',2)
    xlabel('P_{FA}')
    ylabel('P_{CC}')
    legend('In-Situ: After Baseline Training','In-Situ: After In-Situ Learning')
    grid on
    
    subplot(3, 1, 3);hold on
    fa_rate_fin_avg_valid = nanmean(fa_rate_fin_valid);
    cc_rate_fin_avg_valid = nanmean(cc_rate_fin_valid);
    fa_rate_ini_avg_valid = nanmean(fa_rate_ini_valid);
    cc_rate_ini_avg_valid = nanmean(cc_rate_ini_valid);
    plot(fa_rate_ini_avg_valid,cc_rate_ini_avg_valid,'LineWidth',2);
    plot(fa_rate_fin_avg_valid,cc_rate_fin_avg_valid,'r','LineWidth',2)
    xlabel('P_{FA}')
    ylabel('P_{CC}')
    legend('Generalization: After Baseline Training','Generalization: After In-Situ Learning')
    grid on
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(48); clf; hold on
    subplot(3, 1, 1)
    plot( mean(z,2), '--');
    % plot( mean(ztest,2));
    legend('Baseline')
    xlabel('In-Situ Learning Iteration'); ylabel('CCR')
    ylim([.5 1])
    grid minor
    title(['# Partitions: ' num2str(numBatches) ' Batch size: ' num2str(numel(testTempClass))])
    
    subplot(3, 1, 2)
    plot( mean(ztestcheck,2), '--');
    % plot( mean(ztest,2));
    legend('In-Situ')
    xlabel('In-Situ Learning Iteration'); ylabel('CCR')
    ylim([.5 1])
    grid minor
    
    subplot(3, 1, 3)
    plot( mean(zvalidcheck,2), '--');
    % plot( mean(ztest,2));
    legend('Generalization')
    xlabel('In-Situ Learning Iteration'); ylabel('CCR')
    ylim([.5 1])
    grid minor
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure(922); clf;
    subplot(3,1,1)
    plot(sum(changes_baseline, 2)./(size(changes_baseline, 2)), '--')
    grid minor
    subplot(3,1,2)
    plot(sum(changes_test, 2)./(size(changes_test, 2)), '--')
    grid minor
    ylabel('% of samples that have changes in classification')
    subplot(3,1,3)
    plot(sum(changes_valid, 2)./(size(changes_valid, 2)), '--')
    xlabel('In-Situ Learning Iteration')
    grid minor
end
%% plotting that matches the ismkl scripts
figure(348); clf; hold on
plt1=plot(mean(results.class_percent_correct_train, 1),'-.');
plt2=plot(mean(results.class_percent_correct_test, 1),'--');
plt3=plot(mean(results.class_percent_correct_valid, 1));

% [~,i1max]= max(plt1.YData);
% [~,i2max]= max(plt2.YData);
% [~,i3max]= max(plt3.YData);
% makedatatip(plt1,[size(results.correct_class_train,2) i1max ]);
% makedatatip(plt2,[size(results.correct_class_is,2) i2max]);
% makedatatip(plt3,[size(results.correct_class_gen,2) i3max]);

legend('Baseline', 'In-Situ', 'Generalization','location','southeast')
ylim([0, 1])
title(['First IS - Atoms Added: ' num2str(results.atoms_added)])
xlim([1 size(results.class_percent_correct_valid,2)])
xxticks = (xticks-1)*results.batchSize;
xticklabels(xxticks)
xlabel('In-Situ samples learned')
ylabel('Correct Classification Rate');
grid minor

ylim([0 1])
figure(569); clf; hold on

plt1=plot(mean(results2.class_percent_correct_train, 1),'-.');
plt2=plot(mean(results2.class_percent_correct_test, 1),'--');
plt3=plot(mean(results2.class_percent_correct_valid, 1));

% [~,i1max]= max(plt1.YData);
% [~,i2max]= max(plt2.YData);
% [~,i3max]= max(plt3.YData);
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