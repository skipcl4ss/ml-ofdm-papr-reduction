import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ofdm.papr import calculate_papr
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time
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
iterations_papr_dict = {i: [] for i in range(iterations)}

for _ in range(samples_per_L):
    # Generate 16-QAM Symbols
    symbols = qam16_mod(N)

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
        iterations_papr_dict[i].append(calculate_papr(x_current))

# CCDF Calculation and Plotting

# Plot Unclipped CCDF (Baseline)
papr_unclipped_sorted = np.sort(papr_unclipped)
ccdf_unclipped = 1 - np.arange(len(papr_unclipped_sorted)) / len(papr_unclipped_sorted)
mask_un = (ccdf_unclipped >= 0.0001)

plt.figure(figsize=(10, 7))
plt.semilogy(papr_unclipped_sorted[mask_un], ccdf_unclipped[mask_un],
             'b-', label='Unclipped', lw=2)

# Plot Clipped & Filtered CCDFs
for i in range(iterations):
    p_sorted = np.sort(iterations_papr_dict[i])
    ccdf_val = 1.0 - np.arange(len(p_sorted)) / len(p_sorted)
    mask = ccdf_val >= 0.0001
    plt.semilogy(p_sorted[mask], ccdf_val[mask], label=f'ICF ({i + 1} iteration{"s" if i > 0 else ""})', lw=2)

# --- Plot Formatting ---
plt.axhline(y=1e-2, color='gray', linestyle=':', alpha=0.6)
plt.xlim([4, 12])
plt.ylim([1e-4, 1])
plt.title(f'PAPR Distribution: 16QAM (N={N}, L={L}, CR={cr_dB}dB)\nICF')
plt.xlabel('PAPR$_0$ [dB]')
plt.ylabel('Prob(PAPR > PAPR$_0$)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")
