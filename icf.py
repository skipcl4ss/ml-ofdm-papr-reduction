import numpy as np
from ofdm.modem import qam16_mod, qpsk_mod
from ofdm.candf import clip_time, oversample_time, filter_time
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.plots import plot_ccdf_compare, plot_ccdf
from nnicf import NNICFMapper, normalize, denormalize
import torch
from scipy import signal
import os
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

# model_dir = "./trained_models/"
model_dir = "./new architecture/"

# * Load nnicf model

# 2. Load the trained models
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
# NN_Mod_Re = NNICFMapper(N_fft).to(device)
# NN_Mod_Im = NNICFMapper(N_fft).to(device)
NN_Mod_Re = NNICFMapper().to(device)
NN_Mod_Im = NNICFMapper().to(device)

# done: retest all current models after denormalization, didnt matter much tho as the models themselves were flawed from the start
# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_re_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_im_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))
# NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_re_weights_{lr_str}.pth"), weights_only=True))
# NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_im_weights_{lr_str}.pth"), weights_only=True))
# NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"mod_re_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))
# NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"mod_im_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))
# NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"mod_re_weights_{train_test_str}.pth"), weights_only=True))
# NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"mod_im_weights_{train_test_str}.pth"), weights_only=True))
# NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"mod_re_weights.pth"), weights_only=True))
# NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"mod_im_weights.pth"), weights_only=True))

NN_Mod_Re.eval()
NN_Mod_Im.eval()

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

# Simulation
papr_unclipped, cm_unclipped = [], []
iterations_papr = [[] for _ in range(iterations)]
iterations_cm = [[] for _ in range(iterations)]

# todo: implement scf
# todo: plot time domain signal

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

    # Oversample and convert to time domain
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

        # ! filter_time() is surprisingly better
        x_time = filter_time(x_clipped, N)
        # # Filtering: use lfilter (or filtfilt for zero-phase)
        # x_time = signal.lfilter(b, a, x_clipped).astype(np.complex64)
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

middle = time.time()

# * Compare with nnicf model

# 1. Normalize the data and turn it to torch tensors
tx_real, tx_minmax_real = normalize(tx_time[0])
tx_imag, tx_minmax_imag = normalize(tx_time[1])


# 3. Generate Predictions (No gradients needed for testing)
with torch.no_grad():
    # predicted_real = NN_Mod_Re(tx_real).numpy()
    # predicted_imag = NN_Mod_Im(tx_imag).numpy()

    # Memoryless flattening
    X_real_flat = tx_real.view(-1, 1).to(device)
    X_imag_flat = tx_imag.view(-1, 1).to(device)

    predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
    predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()

# predicted_real = torch.tensor(predicted_real, dtype=torch.float32)
# predicted_imag = torch.tensor(predicted_imag, dtype=torch.float32)

# pred_denorm_real = denormalize(predicted_real)
# pred_denorm_imag = denormalize(predicted_imag)
pred_denorm_real = denormalize(predicted_real, tx_minmax_real)
pred_denorm_imag = denormalize(predicted_imag, tx_minmax_imag)

# pred_denorm_real = np.array(pred_denorm_real)
# pred_denorm_imag = np.array(pred_denorm_imag)

# 4. Reconstruct the Complex OFDM Signals
# each has shape of samples_per_L, (N * L)
predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

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

# 1. Define the target y-levels (probabilities)
papr_target_y = 1e-4
cm_target_y = 1e-3

# 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
papr_percentile = (1.0 - papr_target_y) * 100.0
cm_percentile = (1.0 - cm_target_y) * 100.0

# 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
# (This completely replaces the need for y_axis, np.where, and manual sorting!)
papr_vlines = [
    np.percentile(papr_unclipped, papr_percentile),
    np.percentile(iterations_papr[-1], papr_percentile),
    np.percentile(pred_papr, papr_percentile)
]

cm_vlines = [
    np.percentile(cm_unclipped, cm_percentile),
    np.percentile(iterations_cm[-1], cm_percentile),
    np.percentile(pred_cm, cm_percentile)
]

plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)

# labels = ["OG", "Clipped 1", "Clipped 2", "Clipped 3"]
# plot_ccdf_compare([papr_unclipped, *iterations_papr], label=labels, metric="papr")
# plot_ccdf_compare([cm_unclipped, *iterations_cm], label=labels, metric="cm")

end = time.time()
# ~8s
print(f"\nICF execution time: {middle - start:.2f} seconds")
print(f"\nNNICF execution time: {end - middle:.2f} seconds")
print(f"\nTotal execution time: {end - start:.2f} seconds")
