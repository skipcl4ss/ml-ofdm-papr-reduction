% PARR_of_preamble.m
% Plot Fig. 7.10(b) (the PAPR of IEEE802.16e preamble)
clear, clf
N=1024; L=4; Npreamble=114; n=0:Npreamble-1;
for i = 1:Npreamble
    X=load(['.\\Wibro-Preamble\\Preamble_sym' num2str(i-1) '.dat']);
    X = X(:,1); X = sign(X); X = fftshift(X);
    x = IFFT_oversampling(X,N); PAPRdB(i) = PAPR(x);
    x_os = IFFT_oversampling(X,N,L); PAPRdB_os(i) = PAPR(x_os);
end
plot(n,PAPRdB,'-o', n,PAPRdB_os,':*'), title('PAPRdB with oversampling')