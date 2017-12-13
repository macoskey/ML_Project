
load('../Data/train.mat')
n = 1;
for stkIdx = 1:40
    for i = 3:150
        shift(n) = train(stkIdx,i)-train(stkIdx,i-1);
        n = n + 1;
        
    end
end