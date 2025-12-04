function [s,time] = modulation(x,Ts,Nos,Fc)
% modulates x(n*Ts) with carrier frequency Fc and
% Nos-times oversamples for time=[0:Ts/Nos:Nx*Ts-T]
% Ts: Sampling period of x[n]
% Nos: Oversampling factor
% Fc: Carrier frequency
Nx = length(x); offset = 0;
if nargin <5, % Scale and Oversampling period for Baseband
    scale=1; T=Ts/Nos;
else % Scale and Oversampling period for Passband
    scale=sqrt(2); T=1/Fc/2/Nos;
end
t_Ts=[0:T:Ts-T]; time=[0:T:Nx*Ts-T]; % Sampling interval, Whole interval
tmp = 2*pi*Fc*t_Ts+offset; len_Ts=length(t_Ts);
cos_wct = cos(tmp)*scale; sin_wct = sin(tmp)*scale;
for n = 1:Nx
    s((n-1)*len_Ts+1:n*len_Ts) = real(x(n))*cos_wct-imag(x(n))*sin_wct;
end