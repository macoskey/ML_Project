function ts_to_im_separate_stock_market
% 1D time-series to 2D image
% go into the data folder you want to do the operation

clear; clc;

%% load data
load('../Data/synthetic_temporal_data_20171213_152646.mat');
load('../Data/train.mat')
%% train
la = synth_data(:,1); feat = synth_data(:,2:end);
la = la+(1-min(la)); %make them all [1-c]
n_class = max(la); 

for c = 1: n_class
    fprintf('analyzing train class %.1d\n',c)
    [idx, ~] = find(la==c);
    mkdir(['../Data/train/' num2str(c)]);  % make a folder
    for sample = 1:size(idx,1)
        S = generate_recurrence(feat(idx(sample),:),3,4);
        imwrite(S, fullfile('../Data/train', num2str(c), ...
               [sprintf('%02d',sample) '.jpg']));
    end
end

%% Test
la_test = train(:,1); 
la= [la_test]; feat= [train(:,2:end)];
la = la+(1-min(la)); %make them all [1-c]
n_class = max(la); 

for c = 1: n_class
    fprintf('analyzing test class %.1d\n',c)
    [idx, ~] = find(la==c);
    mkdir(['../Data/test/' num2str(c)]);  % make a folder
    for sample = 1:size(idx,1)
        S = generate_recurrence(feat(idx(sample),:),3,4);
        imwrite(S, fullfile('../Data/test', num2str(c), ...
               [sprintf('%02d',sample) '.jpg'])); 
    end
    
end



end

