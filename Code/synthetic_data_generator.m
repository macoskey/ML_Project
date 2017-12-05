% EECS 545 - Final Project
% Training Data Synthesizer
%
% Created: 12.5.17
% 


%% Synthetic data synthesis
% Initialize simulation parameters
N = 90;                     % simulate a 3-month period
M = 1024;                   % number of synthetic stocks
support = rand();
synth_data = zeros(N,M);    % matrix of valid synthetic data
synth_cnt = 1;

price_mu = 0;
price_sigma = 1;
support_mu = 0;
support_sigma = 5;
    
while synth_cnt <= M
    price = zeros(1,N);         % single stock vector
    % Define Gauss parameters for choosing daily change and support value
    
    
    % generate initial price:
    price(1) = normrnd(price_mu,price_sigma);
    
    % Generate support value
    while price(1) < support
        support = normrnd(support_mu,support_sigma);
    end
    
    % Define the number of times it can hit the support
    support_hits = randi([3 8]);
    hit_cntr = 0;
    
    for n = 2:N
        % if the price is at the support, make it go back up
        if abs(price(n)-support) < 0.1
            price(n) = price(n-1) + abs(normrnd(price_mu,price_sigma));
            % if the price is above the support, let it do what it wants
        else
            price(n) = price(n-1) + normrnd(price_mu,price_sigma);
        end
        
        % check if new price went below the support, if so, set it to
        % approximately the support value. Also document how many times it hit
        % the support
        if price(n) < support && hit_cntr < support_hits
            price(n) = support;% + normrnd(0,0.001);
            hit_cntr = hit_cntr + 1;
        end
    end
%     fprintf('support hits: %.1d out of %.1d allowed\n',hit_cntr,support_hits)
%     plot(price)
    
    % save synthetic stock if it meets the desired parameters
    if hit_cntr >=2
        % Make stock prices all positive valued
        synth_data(:,synth_cnt) = price+abs(min(price));
        synth_cnt = synth_cnt + 1;
        fprintf('%.1d synthetic stocks generated\n',synth_cnt-1)
    end
end

%% Show montage of the synthetic data
figure(1)
recurrence_data = zeros(N,N,1,M);
for i = 1:size(synth_data,2)
    X = synth_data(:,i)*synth_data(:,i)';
    recurrence_data(:,:,1,i) = X;
end
q = quantile(recurrence_data(:),256);
montage(recurrence_data,...
    'Size',[32 32],'DisplayRange',[0 q(end)])




























