import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc

def PAPR(x):
    """
    Calculate Peak-to-Average Power Ratio (PAPR)

    Parameters:
    -----------
    x : array_like
        Input signal (can be complex)

    Returns:
    --------
    PAPR_dB : float
        PAPR in dB
    AvgP_dB : float
        Average power in dB
    PeakP_dB : float
        Peak (maximum) power in dB
    """
    x = np.asarray(x)
    Nx = len(x)

    # Extract real and imaginary parts
    xI = np.real(x)
    xQ = np.imag(x)

    # Calculate instantaneous power
    Power = xI ** 2 + xQ ** 2

    # Average power
    AvgP = np.sum(Power) / Nx
    AvgP_dB = 10 * np.log10(AvgP)

    # Peak power
    PeakP = np.max(Power)
    PeakP_dB = 10 * np.log10(PeakP)

    # PAPR
    PAPR_dB = 10 * np.log10(PeakP / AvgP)

    return PAPR_dB, AvgP_dB, PeakP_dB

def mapper(b, N=None):
    """
    Generate PSK/QAM modulated symbols

    Parameters:
    -----------
    b : int
        Number of bits per symbol (modulation order = 2^b)
    N : int, optional
        If provided, generates N random modulated symbols
        Otherwise, generates constellation points for [0:2^b-1]

    Returns:
    --------
    modulated_symbols : ndarray
        Modulated symbols
    Mod : str
        Modulation type string
    """
    M = 2 ** b  # Modulation order

    if b == 1:
        # BPSK
        Mod = 'BPSK'
        A = 1
        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)
        # BPSK: 0 -> -1, 1 -> +1
        modulated_symbols = A * (2 * data - 1)

    elif b == 2:
        # QPSK with pi/4 offset
        Mod = 'QPSK'
        A = 1
        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)
        # QPSK with pi/4 phase offset
        angles = 2 * np.pi * data / M + np.pi / 4
        modulated_symbols = A * np.exp(1j * angles)

    else:
        # QAM
        Mod = f'{M}QAM'
        Es = 1  # Symbol energy
        A = np.sqrt(3 / (2 * (M - 1)) * Es)

        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)

        # Gray-coded QAM constellation
        k = int(np.sqrt(M))  # Square QAM
        I = 2 * (data % k) - k + 1  # In-phase component
        Q = 2 * (data // k) - k + 1  # Quadrature component
        modulated_symbols = A * (I + 1j * Q)

    return modulated_symbols, Mod

def IFFT_oversampling(X, N, L=1):
    """
    Zero-padding and NL-point IFFT for oversampling
    Equivalent to N-point IFFT with interpolation using oversampling factor L

    Parameters:
    -----------
    X : array_like
        Input frequency domain signal (length N)
    N : int
        FFT size
    L : int, optional
        Oversampling factor (default: 1, no oversampling)

    Returns:
    --------
    xt : ndarray
        Time domain signal after oversampling
    time : ndarray
        Time vector
    """
    X = np.asarray(X).flatten()
    NL = N * L
    T = 1 / NL
    time = np.arange(0, 1, T)

    # Zero-padding: insert zeros in the middle of the spectrum
    X_padded = np.concatenate([
        X[:N // 2],
        np.zeros(NL - N),
        X[N // 2:]
    ])

    # NL-point IFFT with scaling by L
    xt = L * np.fft.ifft(X_padded, NL)

    return xt, time

def clipping(x, CR, sigma=None):
    """
    Clip signal based on clipping ratio

    Parameters:
    -----------
    x : array_like
        Input signal
    CR : float
        Clipping Ratio
    sigma : float, optional
        Square root of variance of x. If not provided, it will be calculated.

    Returns:
    --------
    x_clipped : ndarray
        Clipped signal
    sigma : float
        Square root of variance (standard deviation)
    """
    x = np.asarray(x)

    if sigma is None:
        x_mean = np.mean(x)
        x_dev = x - x_mean
        sigma = np.sqrt(np.dot(x_dev, np.conj(x_dev)) / len(x))

    x_clipped = x.copy()
    CL = CR * sigma  # Clipping level

    # Find indices where absolute value exceeds clipping level
    ind = np.abs(x) > CL

    # Clip values: normalize and multiply by clipping level
    x_clipped[ind] = x[ind] / np.abs(x[ind]) * CL

    return x_clipped, sigma

def addcp(x, Ncp):
    """
    Add cyclic prefix to OFDM symbols

    Parameters:
    -----------
    x : array_like
        Input signal (can be 1D or 2D array)
        If 2D, each column is treated as one OFDM symbol
    Ncp : int
        Length of cyclic prefix (number of samples)

    Returns:
    --------
    y : ndarray
        Signal with cyclic prefix added
    """
    x = np.asarray(x)

    # Handle 1D array
    if x.ndim == 1:
        N = len(x)
        cp = x[-Ncp:]
        y = np.concatenate([cp, x])
    else:
        # 2D array (matrix)
        N, num_symbols = x.shape
        y = np.zeros((N + Ncp, num_symbols), dtype=x.dtype)

        for k in range(num_symbols):
            # Take last Ncp samples and prepend them
            cp = x[-Ncp:, k]
            y[:, k] = np.concatenate([cp, x[:, k]])

    return y

def zero_pasting(x):
    """Paste zeros at the center half of the input sequence x"""
    N = len(x)
    M = int(np.ceil(N / 4))
    y = np.concatenate([x[:M], np.zeros(N // 2), x[N - M:]])
    return y


def berawgn_qam(EbN0_dB, M):
    """Analytical BER for QAM in AWGN channel"""
    EbN0 = 10 ** (EbN0_dB / 10)
    k = np.sqrt(M)
    ber = 2 * (1 - 1 / k) / np.log2(M) * 0.5 * erfc(np.sqrt(3 * np.log2(M) / (2 * (M - 1)) * EbN0))
    return ber


# Parameters
SNRdBs = np.arange(0, 11)
N_SNR = len(SNRdBs)
Nblk = 100
CRs = np.arange(0.8, 1.8, 0.2)
N_CR = len(CRs)
gss = '*^<sd'

b = 2
M = 2 ** b
N = 128
Ncp = 0
fs = 1e6
L = 8

Tsym = 1 / (fs / N)
Ts = 1 / (fs * L)
fc = 2e6
wc = 2 * np.pi * fc
t = np.arange(0, 2 * Tsym, Ts) / Tsym

# Normalization factor for QAM (unit average power)
A = 1 / np.sqrt(2)

# Filter design
Fs = 8
Norder = 104
dens = 20
FF = np.array([0, 1.4, 1.5, 2.5, 2.6, Fs / 2])
AA = np.array([0, 0, 1, 1, 0, 0])
WW = np.array([10, 1, 10])
h = signal.remez(Norder + 1, FF, AA[::2], weight=WW, fs=Fs)

# Initialize arrays
CF = np.zeros(Nblk)
CF_c = np.zeros((N_CR, Nblk))
CF_cf = np.zeros((N_CR, Nblk))

ber_analytic = berawgn_qam(SNRdBs - 10 * np.log10(b), M)

# MATLAB: kk1=1:(N/2-Ncp)*L; (1-indexed)
# Python: 0-indexed
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

# Main simulation loop
for i in range(N_SNR):
    SNRdB = SNRdBs[i]

    for ncf in range(3):  # 0: no clip, 1: clip, 2: clip & filter
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
                # Generate random binary data
                msgbin = np.random.randint(0, 2, (b, N))

                # QAM modulation
                msg_serial = msgbin.flatten('F')  # Column-major like MATLAB
                X, _ = mapper(b)
                # Convert bits to symbols
                data_idx = np.zeros(N, dtype=int)
                for k in range(N):
                    data_idx[k] = msgbin[0, k] * 2 + msgbin[1, k]
                X = A * X[data_idx]
                X[0] = 0 + 0j  # DC subcarrier not used

                x, _ = IFFT_oversampling(X, N, L)
                # x has length N*L
                x_b = addcp(x, Ncp * L)
                # When Ncp=0, x_b should still have length N*L
                # x_b_os total = (N/2-Ncp)*L + len(x_b) + N*L/2
                # With Ncp=0: (N/2)*L + N*L + N*L/2 = 2*N*L
                x_b_os = np.concatenate([
                    np.zeros((N // 2 - Ncp) * L, dtype=complex),
                    x_b,
                    np.zeros(N * L // 2, dtype=complex)
                ])
                # Ensure we have the right time vector length
                t_len = len(x_b_os)
                t_use = t[:t_len] if t_len <= len(t) else np.arange(t_len) * Ts / Tsym
                x_p = np.sqrt(2) * np.real(x_b_os * np.exp(1j * 2 * wc * t_use))

                if ncf > 0:
                    x_p_c, _ = clipping(x_p, CR)
                    x_p = x_p_c
                    if ncf > 1:
                        x_p_cf = np.fft.ifft(np.fft.fft(h, len(x_p)) * np.fft.fft(x_p))
                        x_p = np.real(x_p_cf)

                if i == N_SNR - 1:
                    CF[nblk], _, _ = PAPR(x_p)

                # Add AWGN noise
                signal_power = np.mean(x_p[kk2] ** 2)
                noise_power = signal_power / (10 ** (SNRdB / 10))
                noise = np.sqrt(noise_power) * np.random.randn(len(kk2))

                y_p_n = np.concatenate([x_p[kk1], x_p[kk2] + noise, x_p[kk3]])
                t_use_y = t[:len(y_p_n)] if len(y_p_n) <= len(t) else np.arange(len(y_p_n)) * Ts / Tsym
                y_b = np.sqrt(2) * y_p_n * np.exp(-1j * 2 * wc * t_use_y)
                Y_b = np.fft.fft(y_b)
                y_b_z = np.fft.ifft(zero_pasting(Y_b))

                # MATLAB: y_b_t = y_b_z((N/2-Ncp)*L+m+[0:L:(N+Ncp)*L-1]);
                idx = (N // 2 - Ncp) * L + m - 1 + np.arange(0, (N + Ncp) * L, L)
                y_b_t = y_b_z[idx]
                Y_b_f = np.fft.fft(y_b_t[Ncp:], N) * L

                # QAM demodulation
                Y_b_f_norm = Y_b_f / A
                Y_b_bin = np.zeros((b, N), dtype=int)
                Y_b_bin[0, :] = (np.real(Y_b_f_norm) > 0).astype(int)
                Y_b_bin[1, :] = (np.imag(Y_b_f_norm) > 0).astype(int)

                # Count bit errors (excluding DC subcarrier)
                nobe += np.sum(msgbin[:, 1:] != Y_b_bin[:, 1:])

            # Calculate BER
            if ncf == 0:
                ber_no[i] = nobe / Nblk / (N - 1) / b
            elif ncf == 1:
                ber_c[cr, i] = nobe / Nblk / (N - 1) / b
            else:
                ber_cf[cr, i] = nobe / Nblk / (N - 1) / b

            # Calculate CCDF at highest SNR
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

# Plotting
plt.figure(figsize=(12, 5))

# CCDF plot
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

# BER plot
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

print("Simulation completed!")