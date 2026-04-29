import numpy as np
from ofdm.modem import qam16_mod
from ofdm.candf import oversample_time, clip_and_filter_time
from ofdm.metrics import calculate_papr
from ofdm.plots import plot_ccdf_compare
import time

start = time.time()

# Parameters
N = 1024                # Number of Subcarriers
mid = N // 2
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
# ! CP is only needed in ber
# CP = N // 4             # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
clipping_ratios_dB = [1, 3, 5, 7] # Clipping Ratios (CR)
# clipping_ratios = [10 ** (dB/10) for dB in clipping_ratios_dB]  # Clipping Ratios (CR)
clipping_ratios = [10 ** (dB/20) for dB in clipping_ratios_dB]  # Clipping Ratios (CR)

# modulation scheme
mod = "16qam"
M = 16


# Simulation
unclipped_papr = []
clipped_papr_dict = {cr: [] for cr in clipping_ratios}

samples_per_L_minus_cr_loop = 0
cr_loop = 0

for _ in range(samples_per_L):
    t1 = time.time()
    # 1. Generate 16-QAM Symbols
    tx_data = np.random.randint(0, M, N)
    tx_symbols = qam16_mod(tx_data)

    # Oversample and convert to time domain
    x_time = oversample_time(tx_symbols, N, L)

    # Capture Unclipped PAPR
    unclipped_papr.append(calculate_papr(x_time))

    t2 = time.time()
    samples_per_L_minus_cr_loop += t2 - t1
    # Process C&F
    for cr in clipping_ratios:
        x_filtered, _ = clip_and_filter_time(x_time, cr, N)

        # Store PAPR of the result
        clipped_papr_dict[cr].append(calculate_papr(x_filtered))
    t3 = time.time()
    cr_loop += t3 - t2

# CCDF Calculation and Plotting

cr_labels = []
for cr in clipping_ratios_dB:
    cr_labels.append(f'Clipped & Filtered (CR={cr}dB)')

# labels = ['Unclipped', *cr_list]
title = f'PAPR Distribution: 16QAM (N={N}, L={L})\nUnclipped vs. Clipped and filtered at different CRs'
papr_list = [unclipped_papr] + [clipped_papr_dict[cr] for cr in clipping_ratios]

plot_ccdf_compare(papr_list, title, label=['Unclipped'] + cr_labels)

end = time.time()
# ~30
print(f"\nTotal execution time: {end - start:.2f} seconds")
print()
print("time taken in the inner cr loop:", cr_loop)
print("time taken in outer samples_per_L loop:", samples_per_L_minus_cr_loop)
