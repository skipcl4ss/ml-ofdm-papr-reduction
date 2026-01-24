import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

# Parameters
CR = 1.2  # Clipping Ratio
b = 2
N = 128
Ncp = 32
fs = 1e6
L = 8

Tsym = 1 / (fs / N)
Ts = 1 / (fs * L)
fc = 2e6
wc = 2 * np.pi * fc
t = np.arange(0, 2 * Tsym, Ts) / Tsym
t0 = t[(N//2 - Ncp) * L]
f = np.arange(0, L * fs, fs / (N * 2)) - L * fs / 2

# Filter design
Fs = 8
Norder = 104
dens = 20
FF = np.array([0, 1.4, 1.5, 2.5, 2.6, Fs/2])
AA = np.array([0, 0, 1, 1, 0, 0])
WW = np.array([10, 1, 10])
h = signal.remez(Norder + 1, FF, AA[::2], weight=WW, fs=Fs)

# Generate QPSK modulated signal
X, _ = mapper(b, N)
X[0] = 0  # DC subcarrier not used

# IFFT and oversampling
x, _ = IFFT_oversampling(X, N, L)

# Add cyclic prefix
x_b = addcp(x, Ncp * L)

# Oversampling with zeros
x_b_os = np.concatenate([
    np.zeros((N//2 - Ncp) * L),
    x_b,
    np.zeros(N * L // 2)
])

# From baseband to passband
x_p = np.sqrt(2) * np.real(x_b_os * np.exp(1j * 2 * wc * t))

# Clipping
x_p_c, _ = clipping(x_p, CR)

# Filtering
x_p_c_f = signal.lfilter(h, 1, x_p_c)
X_p_c_f = np.fft.fft(x_p_c_f)

# From passband to baseband
x_b_c_f = np.sqrt(2) * x_p_c_f * np.exp(-1j * 2 * wc * t)

# Define indices for plotting
nn = (N//2 - Ncp) * L + np.arange(N * L)
nn1 = N//2 * L + np.arange(-Ncp * L + 1, 1)
nn2 = N//2 * L + np.arange(0, N * L)

# Figure 1: Original signal
fig1 = plt.figure(figsize=(14, 10))

# Subplot 1: Time domain amplitude (baseband oversampled)
plt.subplot(221)
plt.plot(t[nn1] - t0, np.abs(x_b_os[nn1]), 'k:', label='CP')
plt.plot(t[nn2] - t0, np.abs(x_b_os[nn2]), 'k-', label='Data')
plt.xlabel('t (normalized by symbol duration)')
plt.ylabel('abs(x[m])')
plt.title('Baseband Oversampled Signal Amplitude')
plt.legend()
plt.grid(True)

# Subplot 2: PDF of passband signal
plt.subplot(222)
pdf_x_p, bin_edges = np.histogram(x_p[nn], bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title('Unclipped Passband Signal')
plt.grid(True, alpha=0.3)

# Subplot 3: PSD of baseband oversampled
plt.subplot(223)
XdB_p_os = 20 * np.log10(np.abs(np.fft.fft(x_b_os)))
XdB_p_os_shifted = np.fft.fftshift(XdB_p_os) - np.max(XdB_p_os)
plt.plot(f, XdB_p_os_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Baseband Oversampled Signal')
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

# Subplot 4: PSD of passband
plt.subplot(224)
XdB_p = 20 * np.log10(np.abs(np.fft.fft(x_p)))
XdB_p_shifted = np.fft.fftshift(XdB_p) - np.max(XdB_p)
plt.plot(f, XdB_p_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Passband Signal')
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

plt.tight_layout()
plt.savefig('PDF_original_OFDM_signal.png', dpi=150)

# Figure 2: Clipped and filtered signal
fig2 = plt.figure(figsize=(14, 10))

# Subplot 1: PDF of clipped passband signal
plt.subplot(221)
pdf_x_p_c, bin_edges = np.histogram(x_p_c[nn], bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p_c, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title(f'Clipped Passband Signal, CR={CR}')
plt.grid(True, alpha=0.3)

# Subplot 2: PDF of clipped & filtered passband signal
plt.subplot(222)
pdf_x_p_c_f, bin_edges = np.histogram(x_p_c_f[nn], bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p_c_f, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title(f'Passband Signal after Clipping & Filtering, CR={CR}')
plt.grid(True, alpha=0.3)

# Subplot 3: PSD of clipped passband
plt.subplot(223)
XdB_p_c = 20 * np.log10(np.abs(np.fft.fft(x_p_c)))
XdB_p_c_shifted = np.fft.fftshift(XdB_p_c) - np.max(XdB_p_c)
plt.plot(f, XdB_p_c_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Clipped Passband Signal')
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

# Subplot 4: PSD of clipped & filtered passband
plt.subplot(224)
XdB_p_c_f = 20 * np.log10(np.abs(X_p_c_f))
XdB_p_c_f_shifted = np.fft.fftshift(XdB_p_c_f) - np.max(XdB_p_c_f)
plt.plot(f, XdB_p_c_f_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Clipped & Filtered Passband Signal')
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

plt.tight_layout()
plt.savefig('PDF_clipped_filtered_OFDM_signal.png', dpi=150)
plt.show()

print("Plotting completed!")
print(f"Clipping Ratio: {CR}")