% OFDM_signal.m
clear
N=8; b=2; M=2^b; L=16; NL=N*L; T=1/NL; time = [0:T:1-T];
[X,Mod] = mapper(b,N); % A block of N=8 QPSK symbols
X(1)=0+j*0; % with no DC-subcarrier
for i = 1:N
    if i<=N/2, x = ifft([zeros(1,i-1) X(i) zeros(1,NL-i+1)],NL);
        else x = ifft([zeros(1,NL-N+i-1) X(i) zeros(1,N-i)],NL);
    end
    xI(i,:) = real(x); xQ(i,:) = imag(x);
end
sum_xI = sum(xI); sum_xQ = sum(xQ);
figure(1), clf, subplot(311)
plot(time,xI,'k:'), hold on, plot(time,sum_xI,'b'),
ylabel('x_{I}(t)');
subplot(312), plot(time,xQ,'k:'); hold on, plot(time,sum_xQ,'b')
ylabel('x_{Q}(t)');
subplot(313), plot(time,abs(sum_xI+j*sum_xQ),'b'); hold on;
ylabel('|x(t)|'); xlabel('t');
clear('xI'), clear('xQ')
N=2^4; NL=N*L; T=1/NL; time=[0:T:1-T]; Nhist=1e3; N_bin=30;
for k = 1:Nhist
    [X,Mod] = mapper(b,N); % A block of N=16 QPSK symbols
    X(1)=0+j*0; % with no DC-subcarrier
    for i = 1:N
        if (i<= N/2) x=ifft([zeros(1,i-1) X(i) zeros(1,NL-i+1)],NL);
            else x=ifft([zeros(1,NL-N/2+i-N/2-1) X(i) zeros(1,N-i)],NL);
        end
        xI(i,:) = real(x); xQ(i,:) = imag(x);
    end
    HistI(NL*(k-1)+1:NL*k)=sum(xI); HistQ(NL*(k-1)+1:NL*k)=sum(xQ);
end
figure(2), clf
subplot(311), [xId,bin]=hist(HistI,N_bin);
    bar(bin,xId/sum(xId),'k');
title([Mod ', N=' num2str(N)]); ylabel('pdf of x_{I}(t)');
subplot(312), [xQd,bin]=hist(HistQ,N_bin);
    bar(bin,xQd/sum(xQd),'k');
ylabel('pdf of x_{Q}(t)');
subplot(313), [xAd,bin]=hist(abs(HistI+j*HistI),N_bin);
bar(bin,xAd/sum(xAd),'k'); ylabel('pdf of |x(t)|'); xlabel('x_{0}');