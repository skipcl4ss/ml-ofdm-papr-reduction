import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

start = time.time()

# --- Parameters ---
N = 128
L = 8
N_fft = N * L
CP = 32
iterations = 10000
# clipping_ratios = [0.8, 1.0, 1.2, 1.4, 1.6]
clipping_ratios = [10 ** (1/10), 10 ** (3/10), 10 ** (5/10), 10 ** (7/10)]


# --- 1. QPSK Modulation ---
def qpsk_mod(n_symbols):
    bits = np.random.randint(0, 2, (n_symbols, 2))
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    return symbols / np.sqrt(2)


# --- 2. Filter Design (IIR Chebyshev Type I) ---
# Low-Pass Filter to suppress out-of-band clipping noise
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp, btype='low', analog=False)


def calculate_papr(signal_time):
    peak = np.max(np.abs(signal_time) ** 2)
    average = np.mean(np.abs(signal_time) ** 2)
    return 10 * np.log10(peak / average)


# --- Simulation ---
plt.figure(figsize=(10, 7))

# Storage for the unclipped PAPR results
papr_unclipped = []

# Dictionary to store results for different CRs
clipped_papr_dict = {cr: [] for cr in clipping_ratios}

for _ in range(iterations):
    # 1. Generate QPSK Symbols
    symbols = qpsk_mod(N)

    # 2. Oversampling (Zero Padding)
    symbols_oversampled = np.zeros(N_fft, dtype=complex)
    symbols_oversampled[:N // 2] = symbols[:N // 2]
    symbols_oversampled[-N // 2:] = symbols[N // 2:]

    # 3. IFFT to Time Domain (The "Analog" Signal)
    x_time = np.fft.ifft(symbols_oversampled) * L

    # --- Capture Unclipped PAPR ---
    papr_unclipped.append(calculate_papr(x_time))

    # --- Process Clipping and Filtering for each CR ---
    magnitude = np.abs(x_time)
    phase = np.angle(x_time)
    rms = np.sqrt(np.mean(magnitude ** 2))

    for cr in clipping_ratios:
        threshold = cr * rms

        # 4. Polar Hard Clipping
        magnitude_clipped = np.minimum(magnitude, threshold)
        x_clipped = magnitude_clipped * np.exp(1j * phase)

        # 5. Filtering (IIR Chebyshev LPF)
        x_filtered = signal.lfilter(b, a, x_clipped)

        # 6. Store Result
        clipped_papr_dict[cr].append(calculate_papr(x_filtered))

# --- CCDF Calculation and Plotting ---

# --- Updated Unclipped Plotting Block ---
# Sorting and calculating CCDF
papr_unclipped_sorted = np.sort(papr_unclipped)
ccdf_unclipped = 1 - np.arange(len(papr_unclipped_sorted)) / len(papr_unclipped_sorted)

# Set mask to 10^-4 to ensure the curve reaches the 14dB+ region
# before being cut off by the plot limits
mask_un = (ccdf_unclipped >= 0.0001)

# Plotting with the specified black dashed line style
plt.semilogy(papr_unclipped_sorted[mask_un], ccdf_unclipped[mask_un],
             'b-', label='Unclipped (Baseline)', lw=1.5)

# Ensure the X-axis allows the curve to reach 14-16dB
plt.xlim([2, 16])

# Plot Clipped & Filtered CCDFs
for cr in clipping_ratios:
    p_sorted = np.sort(clipped_papr_dict[cr])
    ccdf_val = 1.0 - np.arange(len(p_sorted)) / len(p_sorted)

    mask = ccdf_val >= 0.0001
    plt.semilogy(p_sorted[mask], ccdf_val[mask], label=f'Clipped (CR={cr})')

# --- Plot Formatting ---
plt.axhline(y=1e-2, color='gray', linestyle=':', alpha=0.6)
plt.title(f'CCDF of PAPR: QPSK (N={N}, L={L})\nUnclipped vs. Clipped and filtered & Chebyshev IIR Filtered')
plt.xlabel('PAPR$_0$ [dB]')
plt.ylabel('Prob(PAPR > PAPR$_0$)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

end = time.time()
# ~7s
print(f"\nTotal execution time: {end - start:.2f} seconds")