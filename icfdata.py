import numpy as np
from ofdm.modem import qam16_mod, qpsk_mod
from ofdm.candf import clip_time, oversample_time, filter_time
from scipy import signal
import time
import os
# import gc

start = time.time()

# Parameters
N = 256                 # Number of Subcarriers
mid = N // 2
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
# ? is CP only needed in ber
CP = N // 4             # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3

# modulation scheme
mod = "16qam"
M = 16
# mod = "qpsk"
# M = 4

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

# todo: implement scf

# tx = []
# rx = []
before_loop = time.time()
for i in range(100):
    loop_start = time.time() if i else before_loop

    tx_time, rx_time = [[], []], [[], []]
    for _ in range(samples_per_L):
        tx_data = np.random.randint(0, M, N)
        if mod == "16qam":
            # Generate 16-QAM Symbols
            tx_symbols = qam16_mod(tx_data)
        elif mod == "qpsk":
            # Generate QPSK Symbols
            tx_symbols = qpsk_mod(tx_data)

        # done: use candf.py
        # # Oversampling via Spectral Centering (Crucial for hitting 14dB)
        # tx_symbols_oversampled = np.zeros(N_fft, dtype=np.complex64)
        # tx_symbols_oversampled[:mid] = tx_symbols[:mid]
        # tx_symbols_oversampled[-mid:] = tx_symbols[mid:]
        #
        # # IFFT to Time Domain (Capturing true analog peaks)
        # # Scale by L to maintain power through the zero-padded IFFT
        # x_time = (np.fft.ifft(tx_symbols_oversampled) * L).astype(np.complex64)

        x_time = oversample_time(tx_symbols, N, L)

        # separately store real and imag parts (needed to generate data for NN model training)
        tx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
        tx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))

        # Process C&F
        for j in range(iterations):
            x_clipped = clip_time(x_time, cr)

            # todo: retrain with filter_time()
            x_time = filter_time(x_clipped, N)
            # Filtering: use lfilter (or filtfilt for zero-phase)
            # x_time = signal.lfilter(b, a, x_clipped).astype(np.complex64)
            # x_time = signal.filtfilt(b, a, x_clipped)

            # store the final iteration's real and imag parts separately
            if j == iterations - 1:
                rx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
                rx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))
    tx_time = np.array(tx_time, dtype=np.float32)
    rx_time = np.array(rx_time, dtype=np.float32)

    os.makedirs("data_npz", exist_ok=True)
    # np.savez_compressed(f"data_npz/16qam_tx_rx_32_part_{i:02d}.npz", tx=tx_time, rx=rx_time)
    np.savez_compressed(f"data_npz/qpsk_tx_rx_32_part_{i:02d}.npz", tx=tx_time, rx=rx_time)
    del tx_time, rx_time
    # gc.collect()

    # tx.append(tx_chunk)
    # rx.append(rx_chunk)

    loop_end = time.time()
    print(f"Iteration {i + 1}/100 completed in {loop_end - loop_start:.2f} seconds")
    print(f"{loop_end - before_loop:.2f} seconds passed since before loop start")

end = time.time()
# ~
print(f"\nTotal execution time: {end - start:.2f} seconds")
