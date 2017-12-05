load snp500.mat

for i = 1:499
stocks = hist_stock_data(now-150, now, snp500(i));
cls = stocks.Close;
open = stocks.Open;
high = stocks.High;
low = stocks.Low; 
vol = stocks.Volume;

subplot(2,1,1)
candle(high,low,cls,open)
%plot(close)
title(snp500(i))
subplot(2,1,2)
plot(vol)
pause
end


%%
dist = zeros(length(tmpData),length(tmpData));
for n = 1:length(dist)
   for m = 1:length(dist)
      dist(n,m) = abs(tmpData(n) - tmpData(m)); 
   end
end

figure(10);
imagesc(dist)
axis square

%% 
figure;
plot(tmpData(2:end),diff(tmpData),'o')

%%
[u,s,v] = svd(tmpData.*tmpData');
r = 1;
dsvd = u(:,1:r)*s(1:r,1:r)*v(:,1:r)';
plot(diag(dsvd))

%%
ftdata = fftshift(fft(tmpData));
N = length(tmpData);
fs = 1/86400; % sampling frequency = once every day
f = [-N/2:N/2-1]; % frequency in days

figure(145), clf
plot(f,abs(ftdata))
xlim([0 45]), ylim([0 50])
filt = gaussLPF(10,1,f,1);
hold on
plot(f,filt.*20)

dataFilt = ftdata.*filt';
outD = ifft(fftshift(dataFilt));
figure(1212)
plot(real(outD))












