import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mapper import mapper
from IFFT_oversampling import IFFT_oversampling
from addcp import addcp
from clipping import clipping
from berawgn import berawgn_qam
from PAPR import PAPR
from zero_pasting import zero_pasting

# todo: compare with matlab code and answer the questions in the comments
# * uncolored comments are the original comments found in the matlab code, which comes from the reference book

# CCDF_of_clipped_filtered_OFDM_signal.m
# Plot Fig. 7.16

# * Parameters
SNRdBs = np.arange(11)  # SNR[dB] vector
N_SNR = len(SNRdBs)
Nblk = 100
CRs = np.arange(0.8, 1.8, 0.2)
N_CR = len(CRs)
gss = '*^<sd'

b = 2       # Number of bits per QAM symbol
M = 2 ** b  # Alphabet size
N = 128     # FFT size
Ncp = 0     # CP size (GI length)
fs = 1e6    # Sampling frequency
L = 8       # Oversampling factor

Tsym = 1 / (fs / N) # OFDM symbol period
Ts = 1 / (fs * L)   # Sampling period
fc = 2e6            # Carrier frequency
wc = 2 * np.pi * fc
# ? shouldnt it be 2Tsym - Ts
t = np.arange(0, 2 * Tsym, Ts) / Tsym   # Time vector

# * Normalization factor for QAM (unit average power)
A = 1 / np.sqrt(2)  # Normalization factor

# * Filter design
Fs = 8          # Baseband sampling frequency
Norder = 104    # Order of filter
dens = 20       # Density factor of filter
FF = np.array([0, 1.4, 1.5, 2.5, 2.6, Fs / 2])  # Stopband/Passband/Stopband frequency edge vector
AA = np.array([0, 0, 1, 1, 0, 0])
WW = np.array([10, 1, 10])  # Stopband/Passband/Stopband weight vector
# ? this is clearly different from matlab's firpm, but how
h = signal.remez(Norder + 1, FF, AA[::2], weight=WW, fs=Fs) # BPF coefficients

# * Initialize arrays
CF = np.zeros(Nblk)
CF_c = np.zeros((N_CR, Nblk))
CF_cf = np.zeros((N_CR, Nblk))

ber_analytic = berawgn_qam(SNRdBs - 10 * np.log10(b), M)

# * MATLAB: kk1=1:(N/2-Ncp)*L; (1-indexed)
# * Python: 0-indexed
kk1 = np.arange((N // 2 - Ncp) * L)
kk2 = np.arange(kk1[-1] + 1, N // 2 * L + N * L + 1)
kk3 = np.arange(kk2[-1] + 1, kk2[-1] + 1 + N * L // 2)

z = np.arange(2, 16.1, 0.1)
len_z = len(z)

ber_no = np.zeros(N_SNR)
ber_c = np.zeros((N_CR, N_SNR))
ber_cf = np.zeros((N_CR, N_SNR))
CCDF_no = np.zeros(len_z)
CCDF_c = np.zeros((N_CR, len_z))
CCDF_cf = np.zeros((N_CR, len_z))

# * Main simulation loop
# Iteration with increasing SNRdB
for i in range(N_SNR):
    SNRdB = SNRdBs[i]

    for ncf in range(3):    # * 0: no clip, 1: clip, 2: clip & filter
        if ncf == 2:
            m = int(np.ceil(len(h) / 2))
        else:
            m = 1

        for cr in range(N_CR):
            if ncf == 0 and cr > 0:
                break

            CR = CRs[cr]
            nobe = 0

            for nblk in range(Nblk):
                # * Generate random binary data
                msgbin = np.random.randint(0, 2, (b, N))

                # QAM modulation
                msg_serial = msgbin.flatten('F')    # * Column-major like MATLAB
                # ? why only the first output
                X, _ = mapper(b)
                # * Convert bits to symbols
                data_idx = np.zeros(N, dtype=int)
                for k in range(N):
                    data_idx[k] = msgbin[0, k] * 2 + msgbin[1, k]
                X = A * X[data_idx]
                X[0] = 0 + 0j   # DC subcarrier not used

                # * IFFT and oversampling
                # ? why only the first output
                x, _ = IFFT_oversampling(X, N, L)   # IFFT and oversampling

                # * Add cyclic prefix
                # * x has length N*L
                x_b = addcp(x, Ncp * L) # Add CP
                # * When Ncp=0, x_b should still have length N*L
                # * x_b_os total = (N/2-Ncp)*L + len(x_b) + N*L/2
                # * With Ncp=0: (N/2)*L + N*L + N*L/2 = 2*N*L
                # * Oversampling with zeros
                # ? why use square brackets
                x_b_os = np.concatenate([
                    np.zeros((N // 2 - Ncp) * L, dtype=complex),
                    x_b,
                    np.zeros(N * L // 2, dtype=complex)
                ])  # Oversampling

                # * Ensure we have the right time vector length
                t_len = len(x_b_os)
                t_use = t[:t_len] if t_len <= len(t) else np.arange(t_len) * Ts / Tsym
                # * From baseband to passband
                x_p = np.sqrt(2) * np.real(x_b_os * np.exp(1j * 2 * wc * t_use))

                if ncf > 0:
                    # * Clipping
                    # ? why only the first output
                    x_p_c, _ = clipping(x_p, CR)    # Eq.(7.18)
                    x_p = x_p_c # clipping
                    if ncf > 1:
                        x_p_cf = np.fft.ifft(np.fft.fft(h, len(x_p)) * np.fft.fft(x_p))
                        x_p = np.real(x_p_cf)

                if i == N_SNR - 1:
                    # ? why only the first output
                    CF[nblk], _, _ = PAPR(x_p)

                # * Add AWGN noise
                signal_power = np.mean(x_p[kk2] ** 2)
                noise_power = signal_power / (10 ** (SNRdB / 10))
                noise = np.sqrt(noise_power) * np.random.randn(len(kk2))

                y_p_n = np.concatenate([x_p[kk1], x_p[kk2] + noise, x_p[kk3]])
                t_use_y = t[:len(y_p_n)] if len(y_p_n) <= len(t) else np.arange(len(y_p_n)) * Ts / Tsym
                y_b = np.sqrt(2) * y_p_n * np.exp(-1j * 2 * wc * t_use_y)
                Y_b = np.fft.fft(y_b)
                y_b_z = np.fft.ifft(zero_pasting(Y_b))

                # * MATLAB: y_b_t = y_b_z((N/2-Ncp)*L+m+[0:L:(N+Ncp)*L-1]);
                idx = (N // 2 - Ncp) * L + m - 1 + np.arange(0, (N + Ncp) * L, L)
                y_b_t = y_b_z[idx]
                Y_b_f = np.fft.fft(y_b_t[Ncp:], N) * L

                # * QAM demodulation
                Y_b_f_norm = Y_b_f / A
                Y_b_bin = np.zeros((b, N), dtype=int)
                Y_b_bin[0, :] = (np.real(Y_b_f_norm) > 0).astype(int)
                Y_b_bin[1, :] = (np.imag(Y_b_f_norm) > 0).astype(int)

                # * Count bit errors (excluding DC subcarrier)
                nobe += np.sum(msgbin[:, 1:] != Y_b_bin[:, 1:])
            # End of the nblk loop

            # * Calculate BER
            if ncf == 0:
                ber_no[i] = nobe / Nblk / (N - 1) / b
            elif ncf == 1:
                ber_c[cr, i] = nobe / Nblk / (N - 1) / b
            else:
                ber_cf[cr, i] = nobe / Nblk / (N - 1) / b

            # * Calculate CCDF at highest SNR
            if i == N_SNR - 1:
                CCDF = np.zeros(len_z)
                for iz in range(len_z):
                    CCDF[iz] = np.sum(CF > z[iz]) / Nblk

                if ncf == 0:
                    CCDF_no = CCDF
                    break
                elif ncf == 1:
                    CCDF_c[cr, :] = CCDF
                else:
                    CCDF_cf[cr, :] = CCDF

# * Plotting
plt.figure(figsize=(12, 5))

# * CCDF plot
plt.subplot(121)
plt.semilogy(z, CCDF_no, 'k-', linewidth=2, label='No clipping')
for cr in range(N_CR):
    gs = gss[cr]
    plt.semilogy(z, CCDF_c[cr, :], f'{gs}-', label=f'CR={CRs[cr]:.1f} clipped')
    plt.semilogy(z, CCDF_cf[cr, :], f'{gs}:', label=f'CR={CRs[cr]:.1f} clip&filt')
plt.grid(True)
plt.xlabel('PAPR [dB]')
plt.ylabel('CCDF')
plt.title('CCDF of PAPR')
plt.legend(fontsize=7)

# * BER plot
plt.subplot(122)
for cr in range(N_CR):
    gs = gss[cr]
    plt.semilogy(SNRdBs, ber_c[cr, :], f'{gs}-', label=f'CR={CRs[cr]:.1f} clipped')
    plt.semilogy(SNRdBs, ber_cf[cr, :], f'{gs}:', label=f'CR={CRs[cr]:.1f} clip&filt')
plt.semilogy(SNRdBs, ber_no, 'o-', linewidth=2, label='No clipping')
plt.semilogy(SNRdBs, ber_analytic, 'k-', linewidth=2, label='Analytical')
plt.grid(True)
plt.xlabel('SNR [dB]')
plt.ylabel('BER')
plt.title('Bit Error Rate')
plt.legend(fontsize=7)

plt.tight_layout()
plt.savefig('CCDF_clipped_filtered_OFDM.png', dpi=150)
plt.show()

print("Plotting completed!")
