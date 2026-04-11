import numpy as np
from ofdm.modem import qam16_mod, qpsk_mod
from ofdm.candf import clip_time, oversample_time
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.ccdf import plot_ccdf_compare, plot_ccdf
from nnicf import NNICFMapper, normalize
import torch
from scipy import signal
import time

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

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
lr = 0.001
lr_str = "dot" + str(lr).split(".")[1]

# Data splitting (used only in saving and loading files, not in the actual C&F process)
train_size = 80
test_size = (100 - train_size) or 1
train_test_str = f"{train_size}_{test_size}"

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

# Simulation
papr_unclipped, cm_unclipped = [], []
iterations_papr = [[] for _ in range(iterations)]
iterations_cm = [[] for _ in range(iterations)]

# todo: implement scf

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

    # separately store real and imag parts (needed for NN model plotting)
    tx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
    tx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))

    # Capture Unclipped PAPR
    papr_unclipped.append(calculate_papr(x_time))
    cm_unclipped.append(calculate_cm(x_time))

    # Process C&F
    for i in range(iterations):
        x_clipped = clip_time(x_time, cr)

        # todo: experiment with clip_and_filter_ofdm()
        # Filtering: use lfilter (or filtfilt for zero-phase)
        x_time = signal.lfilter(b, a, x_clipped).astype(np.complex64)
        # x_time = signal.filtfilt(b, a, x_clipped)

        # Store PAPR of the current iterative result
        iterations_papr[i].append(calculate_papr(x_time))
        iterations_cm[i].append(calculate_cm(x_time))

        # store the final iteration's real and imag parts separately
        if i == iterations - 1:
            rx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
            rx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))
tx_time = np.array(tx_time, dtype=np.float32)
rx_time = np.array(rx_time, dtype=np.float32)

# * Compare with nnicf model

# 1. Normalize the data and turn it to torch tensors
tx_real, rx_real = normalize(tx_time, rx_time, "real")
tx_imag, rx_imag = normalize(tx_time, rx_time, "imag")

# 2. Load the trained models
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNICFMapper(N_fft).to(device)
NN_Mod_Im = NNICFMapper(N_fft).to(device)
# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(f"trained_models/{mod}_mod_re_weights_{train_test_str}_{lr_str}.pth", weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(f"trained_models/{mod}_mod_im_weights_{train_test_str}_{lr_str}.pth", weights_only=True))
NN_Mod_Re.eval()
NN_Mod_Im.eval()

# 3. Generate Predictions (No gradients needed for testing)
with torch.no_grad():
    predicted_real = NN_Mod_Re(tx_real).numpy()
    predicted_imag = NN_Mod_Im(tx_imag).numpy()

# todo: denormalize before recombining
# 4. Reconstruct the Complex OFDM Signals
# each has length of samples_per_L
predicted_complex = predicted_real + 1j * predicted_imag

# 5. Calculate PAPR (Peak-to-Average Power Ratio) for CCDF
pred_papr, pred_cm = [], []
for i in range(samples_per_L):
    pred_papr.append(calculate_papr(predicted_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))
pred_papr = np.array(pred_papr)
pred_cm = np.array(pred_cm)

# 6. Plot the CCDF
title = f"NNICF Predicted OFDM\n{mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)\nlr = {lr} ({train_size} Training/{test_size} Testing)"
labels = ['Original OFDM', f'Clipped OFDM ({iterations} iterations)', 'NNICF Predicted OFDM']

plot_ccdf_compare([papr_unclipped, iterations_papr[-1], pred_papr], f"Original vs Clipped vs {title}", labels)
plot_ccdf_compare([cm_unclipped, iterations_cm[-1], pred_cm], f"Original vs Clipped vs {title}", labels, metric="CM")

y_axis = np.arange(samples_per_L, 0, -1) / samples_per_L
papr_floor = np.where(y_axis == 1e-4)[0]
papr_vlines = [np.sort(papr_unclipped)[papr_floor], np.sort(iterations_papr[-1])[papr_floor]]
plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

cm_floor = np.where(y_axis == 1e-3)[0]
cm_vlines = [np.sort(cm_unclipped)[cm_floor], np.sort(iterations_cm[-1])[cm_floor]]
plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)

# labels = ["OG", "CLipped 1", "CLipped 2", "CLipped 3"]
# plot_ccdf_compare([papr_unclipped, *iterations_papr], label=labels, metric="papr")
# plot_ccdf_compare([cm_unclipped, *iterations_cm], label=labels, metric="cm")

end = time.time()
# ~8s
print(f"\nTotal execution time: {end - start:.2f} seconds")
