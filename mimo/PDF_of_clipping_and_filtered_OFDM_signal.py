import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mapper import mapper
from IFFT_oversampling import IFFT_oversampling
from addcp import addcp
from clipping import clipping

# todo: compare with matlab code and answer the questions in the comments
# * uncolored comments are the original comments found in the matlab code, which comes from the reference book

# PDF_of_clipped_and_filtered_OFDM_signal.m
# Plot Figs. 7.14 and 7.15

# * Parameters
CR = 1.2    # Clipping Ratio
b = 2       # Number of bits per QPSK symbol
N = 128     # FFT size
Ncp = 32    # CP size
fs = 1e6    # Sampling frequency
L = 8       # Oversampling factor

Tsym = 1 / (fs / N) # OFDM symbol period
Ts = 1 / (fs * L)   # Sampling period
fc = 2e6            # Carrier frequency
wc = 2 * np.pi * fc
# ? shouldnt it be 2Tsym - Ts
t = np.arange(0, 2 * Tsym, Ts) / Tsym   # Time vector
# ? why use square brackets
t0 = t[(N // 2 - Ncp) * L]
# ? shouldnt it be L*fs - Ts
f = np.arange(0, L * fs, fs / (N * 2)) - L * fs / 2 # Frequency vector

# * Filter design
Fs = 8          # Baseband sampling frequency
Norder = 104    # Order of filter
dens = 20       # Density factor of filter
FF = np.array([0, 1.4, 1.5, 2.5, 2.6, Fs / 2])  # Stopband/Passband/Stopband frequency edge vector
AA = np.array([0, 0, 1, 1, 0, 0])
WW = np.array([10, 1, 10])  # Stopband/Passband/Stopband weight vector
# ? this is clearly different from matlab's firpm, but how
h = signal.remez(Norder + 1, FF, AA[::2], weight=WW, fs=Fs) # BPF coefficients

# * Generate QPSK modulated signal
# ? why only the first output
X, _ = mapper(b, N) # QPSK modulation
X[0] = 0    # DC subcarrier not used

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
    np.zeros((N // 2 - Ncp) * L),
    x_b,
    np.zeros(N * L // 2)
])  # Oversampling

# * From baseband to passband
x_p = np.sqrt(2) * np.real(x_b_os * np.exp(1j * 2 * wc * t))

# * Clipping
# ? why only the first output
x_p_c, _ = clipping(x_p, CR)    # Eq.(7.18)

# * Filtering
# ? is this identical to matlab's filter
x_p_c_f = signal.lfilter(h, 1, x_p_c)
X_p_c_f = np.fft.fft(x_p_c_f)

# * From passband to baseband
# ? ifft was applied to X_p_c_f in matlab to derive x_p_c_f
x_b_c_f = np.sqrt(2) * x_p_c_f * np.exp(-1j * 2 * wc * t)   # From passband to baseband

# * Define indices for plotting
nn = (N // 2 - Ncp) * L + np.arange(N * L)
nn1 = N // 2 * L + np.arange(-Ncp * L + 1, 1)
nn2 = N // 2 * L + np.arange(N * L)

# * Figure 1: Original signal
fig1 = plt.figure(figsize=(14, 10))

# * Subplot 1: Time domain amplitude (baseband oversampled)
plt.subplot(221)
plt.plot(t[nn1] - t0, np.abs(x_b_os[nn1]), 'k:', label='CP')
plt.plot(t[nn2] - t0, np.abs(x_b_os[nn2]), 'k-', label='Data')
plt.xlabel('t (normalized by symbol duration)')
plt.ylabel('abs(x[m])')
plt.title('Baseband Oversampled Signal Amplitude')
plt.legend()
plt.grid(True)

# * Subplot 2: PDF of passband signal
# ? this block as a whole is weird
plt.subplot(222)
pdf_x_p, bin_edges = np.histogram(x_p[nn], bins=50, density=True)
# ? what is bin_centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title('Unclipped Passband Signal')
plt.grid(True, alpha=0.3)

# * Subplot 3: PSD of baseband oversampled
plt.subplot(223)
XdB_p_os = 20 * np.log10(np.abs(np.fft.fft(x_b_os)))
XdB_p_os_shifted = np.fft.fftshift(XdB_p_os) - np.max(XdB_p_os)
plt.plot(f, XdB_p_os_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Baseband Oversampled Signal')
# ? what are the parameters for axis
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

# * Subplot 4: PSD of passband
plt.subplot(224)
XdB_p = 20 * np.log10(np.abs(np.fft.fft(x_p)))
XdB_p_shifted = np.fft.fftshift(XdB_p) - np.max(XdB_p)
plt.plot(f, XdB_p_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Passband Signal')
# ? what are the parameters for axis
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

plt.tight_layout()
plt.savefig('PDF_original_OFDM_signal.png', dpi=150)

# * Figure 2: Clipped and filtered signal
fig2 = plt.figure(figsize=(14, 10))

# * Subplot 1: PDF of clipped passband signal
# ? this block as a whole is weird
plt.subplot(221)
pdf_x_p_c, bin_edges = np.histogram(x_p_c[nn], bins=50, density=True)
# ? what is bin_centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p_c, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title(f'Clipped Passband Signal, CR={CR}')
plt.grid(True, alpha=0.3)

# Subplot 2: PDF of clipped & filtered passband signal
# ? this block as a whole is weird
plt.subplot(222)
pdf_x_p_c_f, bin_edges = np.histogram(x_p_c_f[nn], bins=50, density=True)
# ? what is bin_centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, pdf_x_p_c_f, width=np.diff(bin_edges)[0], color='k', alpha=0.7)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title(f'Passband Signal after Clipping & Filtering, CR={CR}')
plt.grid(True, alpha=0.3)

# * Subplot 3: PSD of clipped passband
plt.subplot(223)
XdB_p_c = 20 * np.log10(np.abs(np.fft.fft(x_p_c)))
XdB_p_c_shifted = np.fft.fftshift(XdB_p_c) - np.max(XdB_p_c)
plt.plot(f, XdB_p_c_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Clipped Passband Signal')
# ? what are the parameters for axis
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

# * Subplot 4: PSD of clipped & filtered passband
plt.subplot(224)
XdB_p_c_f = 20 * np.log10(np.abs(X_p_c_f))
XdB_p_c_f_shifted = np.fft.fftshift(XdB_p_c_f) - np.max(XdB_p_c_f)
plt.plot(f, XdB_p_c_f_shifted, 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD of Clipped & Filtered Passband Signal')
# ? what are the parameters for axis
plt.axis([f[0], f[-1], -100, 0])
plt.grid(True)

plt.tight_layout()
plt.savefig('PDF_clipped_filtered_OFDM_signal.png', dpi=150)
plt.show()

print("Plotting completed!")
print(f"Clipping Ratio: {CR}")
