import numpy as np
from scipy import signal
from ofdm.papr import calculate_papr
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time
from ofdm.ccdf import plot_CCDF, plot_CCDF_compare
import time

start = time.time()

# Parameters
N = 256                 # Number of Subcarriers
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
CP = 32                 # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp, btype='low', analog=False)

# Simulation
papr_unclipped = []
iterations_papr = [[] for _ in range(iterations)]

# todo: implement scf

for _ in range(samples_per_L):
    # todo: implement qpsk
    # Generate 16-QAM Symbols
    symbols = qam16_mod(N)

    # todo: use candf.py
    # Oversampling via Spectral Centering (Crucial for hitting 14dB)
    symbols_oversampled = np.zeros(N_fft, dtype=np.complex64)
    symbols_oversampled[:N // 2] = symbols[:N // 2]
    symbols_oversampled[-N // 2:] = symbols[N // 2:]

    # IFFT to Time Domain (Capturing true analog peaks)
    # Scale by L to maintain power through the zero-padded IFFT
    x_time = (np.fft.ifft(symbols_oversampled) * L).astype(np.complex64)

    # Capture Unclipped PAPR
    papr_unclipped.append(calculate_papr(x_time))

    # Process C&F
    x_current = x_time
    for i in range(iterations):
        x_clipped = clip_time(x_current, cr)

        # Filtering: use lfilter (or filtfilt for zero-phase)
        x_current = signal.lfilter(b, a, x_clipped).astype(np.complex64)
        # x_current = signal.filtfilt(b, a, x_clipped)

        # Store PAPR of the current iterative result
        iterations_papr[i].append(calculate_papr(x_current))

# todo: add CM metric
# CM = 20 * np.log10(np.mean(np.abs(x_time) ** 4) / (np.mean(np.abs(x_time) ** 2) ** 2))

# CCDF Calculation and Plotting
# Plot Unclipped CCDF (Baseline), in addition to Clipped & Filtered CCDFs
# plot_CCDF(papr_unclipped, N)
plot_CCDF_compare([papr_unclipped] + iterations_papr, N, title_suffix=f'PAPR Distribution: 16QAM (N={N}, L={L}, CR={cr_dB}dB)\nICF')

# todo: add BER

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")
