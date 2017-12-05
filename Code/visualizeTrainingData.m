
% load data
a = pwd;
data = load([a(1:end-4),'Data/train.mat']);
train = data.train;
clear a data

order = randperm(size(train,1));
for n = order
    plot(train(n,2:end))
    if train(n,1) == 1; title 'strong support'
    else title 'nothing'; end
    set(gca,'FontSize',16)
    drawnow, pause
end
