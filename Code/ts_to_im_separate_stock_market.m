function ts_to_im_separate_stock_market
% 1D time-series to 2D image
% go into the data folder you want to do the operation

clear; clc;
% Load UCR time-series Archive data (Two_Pattern data with 4 classes; 1st column is labels)
load('synthetic_data_20171206_095856.mat');

%% train
la_train = train(:,1);
la= [la_train]; feat= [train(:,2:end)];
la = la+(1-min(la)); %make them all [1-c]
n_class = max(la); 

for c = 1: n_class
    c
    [indx, c_la] = find(la==c);
    mkdir(['train/' num2str(c)]);  % make a folder
    for sample = 1:size(indx,1)
        S = RP(feat(indx(sample),:), 3,4);
        imwrite(S, fullfile('train', num2str(c), ...
               [sprintf('%02d',sample) '.jpg']));
    end
end

%% Test
la_test = test(:,1); 
la= [la_test]; feat= [test(:,2:end)];
la = la+(1-min(la)); %make them all [1-c]
n_class = max(la); 

for c = 1: n_class
    c
    [indx, c_la] = find(la==c);
    mkdir(['test/' num2str(c)]);  % make a folder
    for sample = 1:size(indx,1)
        S = RP(feat(indx(sample),:),3,4);
        imwrite(S, fullfile('git test', num2str(c), ...
               [sprintf('%02d',sample) '.jpg'])); 
    end
    
end


function S = RP(s, m, tau)

    % S = 1 X N signal

    y = s';

    N = length(y);
    N2 = N - tau * (m - 1);

    for mi = 1:m;
        xe(:, mi) = y([1:N2] + tau * (mi-1));
    end


    x1 = repmat(xe, N2, 1);
    x2 = reshape(repmat(xe(:), 1, N2)', N2 * N2, m);

    S = sqrt(sum( (x1 - x2) .^ 2, 2 ));
    S = reshape(S, N2, N2);

    imagesc(S)
    %colormap([1 1 1;0 0 0])
    %xlabel('Time (sec)'), ylabel('Time (sec)')
end
end

