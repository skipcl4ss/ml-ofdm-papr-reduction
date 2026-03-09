import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ofdm.papr import calculate_papr
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time
import time

start = time.time()

# --- Parameters (Section 9.1) ---
N = 1024  # Number of Subcarriers
L = 4  # Oversampling Factor
N_fft = N * L  # IFFT Size (extended to 1024)
CP = 32  # Cyclic Prefix
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

tx = []
rx = []
for i in range(4):
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
        for j in range(iterations):
            if j == 0:
                x_current = x_time.copy()

            x_clipped = clip_time(x_current, cr)

            # Filtering: use lfilter (or filtfilt for zero-phase)
            x_current = signal.lfilter(b, a, x_clipped)
            # x_current = signal.filtfilt(b, a, x_clipped)  # uncomment for zero-phase

            # Store PAPR of the current iterative result
            iterations_papr_dict[j].append(calculate_papr(x_current))

            # ! Store the final iteration's real and imag parts separately
            if j == iterations - 1:
                rx_time[0].append(np.real(x_current))
                rx_time[1].append(np.imag(x_current))
    tx.append(tx_time)
    rx.append(rx_time)
tx = np.array(tx)
rx = np.array(rx)

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")