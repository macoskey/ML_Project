function auto_synthetic_data_generator_obvious
% EECS 545 - Final Project
% Training Data Synthesizer
%
% Created: 12.13.17
% 

home = pwd;

N = 150;                     % simulate a 3-month period
M = 512;                   % number of synthetic stocks
support = rand();
synth_data_C1 = zeros(1+N,M);    % matrix of valid synthetic data
synth_data_C1(1,:) = 1;
synth_cnt = 1;

price_mu = 0.01; % assume generally good markets
price_sigma = 1.33;
support_mu = 0;
support_sigma = 5;

%%%% Generate Support Class (C1) Data %%%%
fprintf('Generating support class data...\n')
while synth_cnt <= M
    price = zeros(1,N);         % single stock vector    
    
    % generate initial price:
    price(1) = normrnd(price_mu,price_sigma);
    % Generate support value
    while price(1) < support
        support = normrnd(support_mu,support_sigma);
    end
    
    % Define the number of times it can hit the support
    support_hits = randi([5 10]);
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
            price(n) = support + normrnd(0,0.01);
            hit_cntr = hit_cntr + 1;
        end
    end
%     fprintf('support hits: %.1d out of %.1d allowed\n',hit_cntr,support_hits)
%     plot(price)
    
    % save synthetic stock if it meets the desired parameters
    if hit_cntr >=5
        % Make stock prices all positive valued
        synth_data_C1(2:end,synth_cnt) = price+abs(min(price));
        synth_cnt = synth_cnt + 1;
%         fprintf('%.1d synthetic stocks generated\n',synth_cnt-1)
    end
end

%%%% Generate Non-Support Class Synthetic Data %%%%
synth_data_C2 = zeros(1+N,M);    % matrix of valid synthetic data
synth_data_C2(1,:) = 2;
price_mu = 0;
price_sigma = 1;

fprintf('Generating non-support class data...\n')
for m = 1:M
    price = zeros(1,N);         % single stock vector
    price(1) = normrnd(price_mu,price_sigma);

    for n = 2:N
        price(n) = price(n-1) + normrnd(price_mu,price_sigma);
    end
 
    % Make stock prices all positive valued
    synth_data_C2(2:end,m) = price+abs(min(price));
%     fprintf('%.1d synthetic stocks generated\n',m)
end
fprintf('saving... ')
% Save the data
c = clock();
str = sprintf('synthetic_data_%.4d%.2d%.2d_%.2d%.2d%.0f.mat',...
    c(1),c(2),c(3),c(4),c(5),c(6));
synth_data = [synth_data_C1 synth_data_C2]';

cd ../Data
save(str,'synth_data')
cd(home)
fprintf('saved!\n')
end





















