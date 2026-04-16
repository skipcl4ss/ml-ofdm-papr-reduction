import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ofdm.candf import oversample_time, clip_time, filter_time
from ofdm.plots import plot_ccdf_compare
from ofdm.modem import qam16_mod
from ofdm.metrics import calculate_papr
import time

start = time.time()

# --- Parameters (Section 9.1) ---
N = 1024            # Number of Subcarriers
mid = N // 2
L = 4               # Oversampling Factor
N_fft = N * L       # IFFT Size (extended to 1024)
# CP = 32             # Cyclic Prefix
CP = N // 4             # Cyclic Prefix
samples_per_L = 20000  # High samples_per_L to capture the 14dB tail
# clipping_ratios = [0.8, 1.0, 1.2, 1.4, 1.6] # Clipping Ratios (CR)
# clipping_ratios = [10 ** (1/10), 10 ** (3/10), 10 ** (5/10), 10 ** (7/10)] # Clipping Ratios (CR)
clipping_ratios_dB = [1, 3, 5, 7]
clipping_ratios = [10 ** (1/20), 10 ** (3/20), 10 ** (5/20), 10 ** (7/20)] # Clipping Ratios (CR)

# # --- 1. 16-QAM Modulation (Section 9.2) ---
# def qam16_mod(n_symbols):
#     # Generate random integers 0-15
#     data = np.random.randint(0, 16, n_symbols)
#     # Define 16-QAM mapping
#     mapping = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
#                         -1-3j, -1-1j, -1+3j, -1+1j,
#                          3-3j,  3-1j,  3+3j,  3+1j,
#                          1-3j,  1-1j,  1+3j,  1+1j])
#     symbols = mapping[data]
#     # Normalize power to 1 (Average power of this 16-QAM constellation is 10)
#     return symbols / np.sqrt(10)

# --- 2. Filter Design (Section 9.3: Chebyshev Type I) ---
# IIR Low-Pass Filter design
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

# def calculate_papr(signal_time):
#     peak = np.max(np.abs(signal_time) ** 2)
#     average = np.mean(np.abs(signal_time) ** 2)
#     return 10 * np.log10(peak / average)

# --- Simulation ---
papr_unclipped = []
clipped_papr_dict = {cr: [] for cr in clipping_ratios}

for _ in range(samples_per_L):
    # 1. Generate 16-QAM Symbols
    # symbols = qam16_mod(N)
    tx_data = np.random.randint(0, 16, N)
    symbols = qam16_mod(tx_data)

    # # 2. Oversampling via Spectral Centering (Crucial for hitting 14dB)
    # symbols_oversampled = np.zeros(N_fft, dtype=complex)
    # symbols_oversampled[:N // 2] = symbols[:N // 2]
    # symbols_oversampled[-N // 2:] = symbols[N // 2:]
    #
    # # 3. IFFT to Time Domain (Capturing true analog peaks)
    # # Scale by L to maintain power through the zero-padded IFFT
    # x_time = np.fft.ifft(symbols_oversampled) * L

    x_time = oversample_time(symbols, N, L)

    # --- Capture Unclipped PAPR ---
    papr_unclipped.append(calculate_papr(x_time))

    # --- Process Clipping and Filtering (Section 9.2 & 9.3) ---
    # magnitude = np.abs(x_time)
    # phase = np.angle(x_time)
    # rms = np.sqrt(np.mean(magnitude ** 2))

    for cr in clipping_ratios:
        # threshold = cr * rms
        #
        # # 4. Polar Hard Clipping
        # magnitude_clipped = np.minimum(magnitude, threshold)
        # x_clipped = magnitude_clipped * np.exp(1j * phase)

        x_clipped = clip_time(x_time, cr)

        # done: filter_time() is slightly better than lfilter, probable because this code is of one iteration as opposed to icf
        x_filtered = filter_time(x_clipped, N)
        # # 5. Filtering (The Proposed IIR Filter)
        # x_filtered = signal.lfilter(b, a, x_clipped)

        # 6. Store Result
        clipped_papr_dict[cr].append(calculate_papr(x_filtered))

# --- CCDF Calculation and Plotting ---

cr_list = []
for cr in clipping_ratios_dB:
    cr_list.append(f'Clipped & Filtered (CR={cr}dB)')
labels = ['Unclipped', *cr_list]
title = f'PAPR Distribution: 16QAM (N={N}, L={L})\nUnclipped vs. Clipped and filtered at different CRs'
plot_ccdf_compare([papr_unclipped] + [clipped_papr_dict[cr] for cr in clipping_ratios], title, label=labels)

# # Plot Unclipped CCDF (Baseline)
# papr_unclipped_sorted = np.sort(papr_unclipped)
# ccdf_unclipped = 1 - np.arange(len(papr_unclipped_sorted)) / len(papr_unclipped_sorted)
# mask_un = (ccdf_unclipped >= 0.0001)
#
# plt.figure(figsize=(10, 7))
# plt.semilogy(papr_unclipped_sorted[mask_un], ccdf_unclipped[mask_un],
#              'b-', label='Unclipped', lw=2)
#
# # Plot Clipped & Filtered CCDFs
# for i, cr in enumerate(clipping_ratios):
#     p_sorted = np.sort(clipped_papr_dict[cr])
#     ccdf_val = 1.0 - np.arange(len(p_sorted)) / len(p_sorted)
#     mask = ccdf_val >= 0.0001
#     plt.semilogy(p_sorted[mask], ccdf_val[mask], label=f'Clipped & Filtered (CR={clipping_ratios_dB[i]}dB)')
#
# # --- Plot Formatting ---
# plt.axhline(y=1e-2, color='gray', linestyle=':', alpha=0.6)
# plt.xlim([4, 12]) # Set X-axis to 16dB to match reference image
# plt.ylim([1e-3, 1])
# plt.title(f'PAPR Distribution: 16QAM (N={N}, L={L})\nUnclipped vs. Clipped and filtered at different CRs')
# plt.xlabel('PAPR$_0$ [dB]')
# plt.ylabel('Prob(PAPR > PAPR$_0$)')
# plt.grid(True, which="both", ls="-", alpha=0.3)
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()

end = time.time()
# ~30
print(f"\nTotal execution time: {end - start:.2f} seconds")
