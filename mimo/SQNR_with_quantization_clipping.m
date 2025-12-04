% SQNR_with_quantization_clipping.m
% Plot Fig. 7.12
clear, clf
N=64; b=6; % FFT size, Number of bits per QAM symbol
L=8; MaxIter=10000; % Oversampling factor, Maximum number of iterations
TWLs = [6:9]; IWL = 1; % Total WordLengths and Integral WordLength
mus=[2:0.2:8]; sq2=sqrt(2); % Clipping Ratio vector
sigma=1/sqrt(N); % Variance of x
gss=['ko-';'ks-';'k^-';'kd-']; % Graphic symbols
for i = 1:length(TWLs)
    TWL = TWLs(i); FWL = TWL-IWL; % Total/Fractional WordLength
    for m = 1:length(mus)
        mu = mus(m)/sq2; % To make the real & imaginary parts of x
        % normalized to 1 (not 1/sqrt(2)) in fi() command below
        Tx = 0; Te = 0;
        for k = 1:MaxIter
            X = mapper(b,N); x = ifft(X,N);
            x = x/sigma/mu;
            xq=fi(x,1,TWL,FWL,'roundmode','round','overflowmode','saturate');
            % Note that the fi() command with TWL=FWL+1 (fraction+sign bits,
            % no integer bits) performs clipping as well as quantization
            xq=double(xq); Px = x*x'; e = x-xq; Pe = e*e';
            Tx=Tx+Px; Te=Te+Pe; % Sum of Signal power and Quantization power
        end
        SQNRdB(i,m) = 10*log10(Tx/Te);
    end
end
for i=1:size(gss,1), plot(mus,SQNRdB(i,:),gss(i,:)), hold on; end
xlabel('mus(clipping level normalized to sigma)'); ylabel('SQNR[dB]');