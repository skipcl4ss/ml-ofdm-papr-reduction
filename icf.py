import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ofdm.papr import calculate_papr
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time
import time

start = time.time()

# --- Parameters (Section 9.1) ---
N = 1024            # Number of Subcarriers
L = 4               # Oversampling Factor
N_fft = N * L       # IFFT Size (extended to 1024)
CP = 32             # Cyclic Prefix
samples_per_L = 10000  # High iterations to capture the 14dB tail
cr_dB = 3
cr = 10 ** (cr_dB / 20)

# --- 2. Filter Design (Section 9.3: Chebyshev Type I) ---
# IIR Low-Pass Filter design
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp, btype='low', analog=False)

# --- Simulation ---
plt.figure(figsize=(10, 7))

papr_unclipped = []
iterations = 4
iterations_papr_dict = {i: [] for i in range(iterations)}

tx_time = [[], []]
rx_time = [[], []]
for _ in range(samples_per_L):
    # 1. Generate 16-QAM Symbols
    symbols = qam16_mod(N)

    # 2. Oversampling via Spectral Centering (Crucial for hitting 14dB)
    symbols_oversampled = np.zeros(N_fft, dtype=complex)
    symbols_oversampled[:N // 2] = symbols[:N // 2]
    symbols_oversampled[-N // 2:] = symbols[N // 2:]

    # 3. IFFT to Time Domain (Capturing true analog peaks)
    # Scale by L to maintain power through the zero-padded IFFT
    x_time = np.fft.ifft(symbols_oversampled) * L

    # ! separately store real and imag parts (needed to generate data for NN model training)
    tx_time[0].append(np.real(x_time))
    tx_time[1].append(np.imag(x_time))

    # --- Capture Unclipped PAPR ---
    papr_unclipped.append(calculate_papr(x_time))

    # --- Process Clipping and Filtering (Section 9.2 & 9.3) ---
    for i in range(iterations):
        if i == 0:
            x_current = x_time.copy()

        x_clipped = clip_time(x_current, cr)

        # Filtering: use lfilter (or filtfilt for zero-phase)
        x_current = signal.lfilter(b, a, x_clipped)
        # x_current = signal.filtfilt(b, a, x_clipped)  # uncomment for zero-phase

        # Store PAPR of the current iterative result
        iterations_papr_dict[i].append(calculate_papr(x_current))

        # ! Store the final iteration's real and imag parts separately
        if i == iterations - 1:
            rx_time[0].append(np.real(x_current))
            rx_time[1].append(np.imag(x_current))
tx_time = np.array(tx_time)
rx_time = np.array(rx_time)

# --- CCDF Calculation and Plotting ---

# Plot Unclipped CCDF (Baseline)
papr_unclipped_sorted = np.sort(papr_unclipped)
ccdf_unclipped = 1 - np.arange(len(papr_unclipped_sorted)) / len(papr_unclipped_sorted)
mask_un = (ccdf_unclipped >= 0.0001)

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
plt.xlim([4, 12]) # Set X-axis to 16dB to match reference image
plt.ylim([1e-5, 1])
plt.title(f'PAPR Distribution: 16QAM (N={N}, L={L}, CR={cr_dB})\nICF')
plt.xlabel('PAPR$_0$ [dB]')
plt.ylabel('Prob(PAPR > PAPR$_0$)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")