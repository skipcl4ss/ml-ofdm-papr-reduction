function [xt,time]=IFFT_oversampling(X,N,L)
% Zero-padding (Eq.(7.15)) & NL-point IFFT
% => N-point IFFT & interpolation with oversampling factor L
if nargin<3, L=1; end % Virtually, no oversampling
NL=N*L; T=1/NL; time = [0:T:1-T]; X = X(:).';
xt = L*ifft([X(1:N/2) zeros(1,NL-N) X(N/2+1:end)], NL);