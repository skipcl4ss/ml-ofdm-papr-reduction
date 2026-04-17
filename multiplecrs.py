import numpy as np
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time, oversample_time, filter_time
from ofdm.metrics import calculate_papr
from ofdm.plots import plot_ccdf_compare
from scipy import signal
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

# modulation scheme
mod = "16qam"
M = 16

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)


# --- Simulation ---
unclipped_papr = []
clipped_papr_dict = {cr: [] for cr in clipping_ratios}

for _ in range(samples_per_L):
    # 1. Generate 16-QAM Symbols
    tx_data = np.random.randint(0, M, N)
    tx_symbols = qam16_mod(tx_data)


    # Oversample and convert to time domain
    x_time = oversample_time(tx_symbols, N, L)

    # Capture Unclipped PAPR
    unclipped_papr.append(calculate_papr(x_time))

    # Process C&F
    for cr in clipping_ratios:
        # Clipping
        x_clipped = clip_time(x_time, cr)

        # done: filter_time() is slightly better than lfilter, probable because this code is of one iteration as opposed to icf
        x_filtered = filter_time(x_clipped, N)
        # # Filtering: use lfilter
        # x_filtered = signal.lfilter(b, a, x_clipped)

        # Store PAPR of the Result
        clipped_papr_dict[cr].append(calculate_papr(x_filtered))

# --- CCDF Calculation and Plotting ---

cr_list = []
for cr in clipping_ratios_dB:
    cr_list.append(f'Clipped & Filtered (CR={cr}dB)')

labels = ['Unclipped', *cr_list]
title = f'PAPR Distribution: 16QAM (N={N}, L={L})\nUnclipped vs. Clipped and filtered at different CRs'
papr_list = [unclipped_papr] + [clipped_papr_dict[cr] for cr in clipping_ratios]

plot_ccdf_compare(papr_list, title, label=labels)

end = time.time()
# ~30
print(f"\nTotal execution time: {end - start:.2f} seconds")
