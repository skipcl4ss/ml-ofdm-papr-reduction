% PAPR_of_Chu.m
% Plot Fig. 7.10(a)
clear, clf
N=16; L=4; i=[0:N-1]; k=3; X = exp(j*k*pi/N*(i.*i)); % Eq.(7.17)
[x,time] = IFFT_oversampling(X,N); PAPRdB = PAPR(x);
[x_os,time_os] = IFFT_oversampling(X,N,L); PAPRdB_os = PAPR(x_os);
plot(time,abs(x),'o', time_os,abs(x_os),'k:*')
PAPRdB_without_and_with_oversampling=[PAPRdB PAPRdB_os]