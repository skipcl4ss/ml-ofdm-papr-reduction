% PDF_of_clipped_and_filtered_OFDM_signal.m
% Plot Figs. 7.14 and 7.15
clear
CR = 1.2; % Clipping Ratio
b=2; N=128; Ncp=32; % Number of bits per QPSK symbol, FFT size, CP size
fs=1e6; L=8; % Sampling frequency and oversampling factor
Tsym=1/(fs/N); Ts=1/(fs*L); % Sampling frequency and sampling period
fc=2e6; wc=2*pi*fc; % Carrier frequency
t=[0:Ts:2*Tsym-Ts]/Tsym; t0=t((N/2-Ncp)*L); % Time vector
f=[0:fs/(N*2):L*fs-fs/(N*2)]-L*fs/2; % Frequency vector
Fs=8; Norder=104; dens=20; % Sampling frequency and Order of filter
dens=20; % Density factor of filter
FF=[0 1.4 1.5 2.5 2.6 Fs/2]; % Stop/Pass/Stop frequency edge vector
WW=[10 1 10]; % Stopband/Passband/Stopband weight vector
h=firpm(Norder,FF/(Fs/2),[0 0 1 1 0 0],WW,{dens}); %BPF coefficients
X = mapper(b,N); X(1) = 0; % QPSK modulation
x =IFFT_oversampling(X,N,L); % IFFT and oversampling
x_b = addcp(x,Ncp*L); % Add CP
x_b_os=[zeros(1,(N/2-Ncp)*L), x_b, zeros(1,N*L/2)]; % Oversampling
x_p = sqrt(2)*real(x_b_os.*exp(j*2*wc*t)); % From baseband to passband
x_p_c = clipping(x_p,CR); % Eq.(7.18)
X_p_c_f= fft(filter(h,1,x_p_c));
x_p_c_f = ifft(X_p_c_f);
x_b_c_f = sqrt(2)*x_p_c_f.*exp(-j*2*wc*t); % From passband to baseband

figure(1); clf
nn=(N/2-Ncp)*L+[1:N*L];
nn1=N/2*L+[-Ncp*L+1:0]; nn2=N/2*L+[0:N*L];

subplot(221)
plot(t(nn1)-t0,abs(x_b_os(nn1)),'k:'); hold on;
plot(t(nn2)-t0,abs(x_b_os(nn2)),'k-');
xlabel('t (normalized by symbol duration)'); ylabel('abs(x"[m])');
subplot(223)
XdB_p_os = 20*log10(abs(fft(x_b_os)));
plot(f,fftshift(XdB_p_os)-max(XdB_p_os),'k');
xlabel('frequency[Hz]'); ylabel('PSD[dB]'); axis([f([1 end]) -100 0]);

subplot(222)
[pdf_x_p,bin]=hist(x_p(nn),50); bar(bin,pdf_x_p/sum(pdf_x_p),'k');
xlabel('x'); ylabel('pdf'); title(['Unclipped passband signal']);
subplot(224), XdB_p = 20*log10(abs(fft(x_p)));
plot(f,fftshift(XdB_p)-max(XdB_p),'k');
xlabel('frequency[Hz]'); ylabel('PSD[dB]'); axis([f([1 end]) -100 0]);

figure(2); clf
subplot(221), [pdf_x_p_c,bin]=hist(x_p_c(nn),50);
bar(bin,pdf_x_p_c/sum(pdf_x_p_c),'k'); xlabel('x'); ylabel('pdf');
title(['Clipped passband signal, CR=' num2str(CR)]);
subplot(223), XdB_p_c = 20*log10(abs(fft(x_p_c)));
plot(f,fftshift(XdB_p_c)-max(XdB_p_c),'k');
xlabel('frequency[Hz]'); ylabel('PSD[dB]'); axis([f([1 end]) -100 0]);

subplot(222), [pdf_x_p_c_f,bin] = hist(x_p_c_f(nn),50);
bar(bin,pdf_x_p_c_f/sum(pdf_x_p_c_f),'k');
title(['Passband signal after clipping & filtering, CR=' num2str(CR)]);
xlabel('x'); ylabel('pdf');
subplot(224), XdB_p_c_f = 20*log10(abs(X_p_c_f));
plot(f,fftshift(XdB_p_c_f)-max(XdB_p_c_f),'k');
xlabel('frequency[Hz]'); ylabel('PSD[dB]'); axis([f([1 end]) -100 0]);