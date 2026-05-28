import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, clip_and_filter_time
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.plots import plot_ccdf_compare, plot_ccdf, plot_signals, plot_signals2
import torch
from nnscf import NNSCFMapper, normalize, denormalize
import os
import time

# todo: reconsider whether the normalization is correctly implemented here and in other scripts

start = time.time()

# Parameters
N = 256                 # Number of Subcarriers
mid = N // 2
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
# ! CP is only needed in ber
# CP = N // 4             # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3

# modulation scheme
mod = "16qam"
# mod = "qpsk"
if mod == "16qam":
    M = 16
elif mod == "qpsk":
    M = 4
modulate, _ = get_modem(M)

# clipping technique
tech = "icf"
# tech = "scf"

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
lr = 0.001
# lr = 0.01
lr_str = "dot" + str(lr).split(".")[1]

# Data splitting (used only in saving and loading files, not in the actual C&F process)
datasize = 100
# datasize = 120
train_size = 70
# val_size = 20
val_size = 10
test_size = datasize - train_size - val_size
if test_size:
    one_batch = None
else:
    test_size = 1
    one_batch = "00"
train_val_test_str = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"

# Optimizer
opt = "Adam"
# opt = "LBFGS"

params = f"{opt} optimizer {tech.upper()} {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)\nlr = {lr} ({train_size} Training/{val_size} Validation/{test_size} Testing)"

# model_dir = "./trained_models/"
model_dir = "./new architecture/"

# * Load nnscf model

# 2. Load the trained models
# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNSCFMapper().to(device)
NN_Mod_Im = NNSCFMapper().to(device)

# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_{tech}_mod_re_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_{tech}_mod_im_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))

NN_Mod_Re.eval()
NN_Mod_Im.eval()


# Simulation
unclipped_papr, unclipped_cm = [], []
icf_papr = [[] for _ in range(iterations)]
icf_cm = [[] for _ in range(iterations)]

samples_per_L_minus_icf_time = 0
icf_time = 0

x_time, x_clip, x_filt = [], [], []

tx_time, rx_time = [[], []], [[], []]
for _ in range(samples_per_L):
    t1 = time.time()
    tx_data = np.random.randint(0, M, N)
    tx_symbols = modulate(tx_data)

    # Oversample and convert to time domain
    x_time = oversample_time(tx_symbols, N, L)

    # separately store real and imag parts (needed for NN model plotting)
    tx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
    tx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))

    # Capture Unclipped PAPR
    # unclipped_papr.append(calculate_papr(x_time))
    unclipped_cm.append(calculate_cm(x_time))

    t2 = time.time()
    samples_per_L_minus_icf_time += t2 - t1

    # Process C&F
    x_filt = x_time.copy()
    for i in range(iterations):
        x_filt, x_clip = clip_and_filter_time(x_filt, cr, N)

        # Store PAPR of the current iterative result
        icf_papr[i].append(calculate_papr(x_filt))
        icf_cm[i].append(calculate_cm(x_filt))

    t3 = time.time()
    icf_time += t3 - t2

    # store the final iteration's real and imag parts separately
    rx_time[0].append(np.real(x_filt).astype(np.float32, copy=False))
    rx_time[1].append(np.imag(x_filt).astype(np.float32, copy=False))
    t4 = time.time()
    samples_per_L_minus_icf_time += t4 - t1

tx_time = np.array(tx_time, dtype=np.float32)
rx_time = np.array(rx_time, dtype=np.float32)

signals = {
    'original': x_time,
    'clipped': x_clip,
    'filtered': x_filt
}

middle = time.time()

# * Compare with nnscf model

# 1. Normalize the data and turn it to torch tensors
tx_real, tx_minmax_real = normalize(tx_time[0])
tx_imag, tx_minmax_imag = normalize(tx_time[1])


# 3. Generate Predictions (No gradients needed for testing)
with torch.no_grad():
    # Memoryless flattening
    X_real_flat = tx_real.view(-1, 1).to(device)
    X_imag_flat = tx_imag.view(-1, 1).to(device)

    predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
    predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()


pred_denorm_real = denormalize(predicted_real, tx_minmax_real)
pred_denorm_imag = denormalize(predicted_imag, tx_minmax_imag)


# 4. Reconstruct the Complex OFDM Signals
# each has shape of samples_per_L, (N * L)
predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

signals["predicted"] = predicted_complex[-1]

# 5. Calculate PAPR (Peak-to-Average Power Ratio) for CCDF
pred_papr, pred_cm = [], []
for i in range(samples_per_L):
    # pred_papr.append(calculate_papr(predicted_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))
pred_papr = np.array(pred_papr)
pred_cm = np.array(pred_cm)

# 6. Plot the CCDF
title = f"NN{tech.upper()} Predicted OFDM\n{params}"
labels = ['Original', f'ICF ({iterations} iteration{"s" if iterations > 1 else ""})', f'NN{tech.upper()} Predicted']
papr_list = [unclipped_papr, icf_papr[-1], pred_papr]
cm_list = [unclipped_cm, icf_cm[-1], pred_cm]

# plot_ccdf_compare(papr_list, f"Original vs {tech.upper()} vs {title}", labels)
plot_ccdf_compare(cm_list, f"Original vs {tech.upper()} vs {title}", labels, metric="CM")

# fixme: both percentile and max are not the most efficient solutions

# todo: find a way to embed the floor part into the plotting function
# 1. Define the target y-levels (probabilities)
papr_target_y = 1e-4
cm_target_y = 1e-3

# 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
papr_percentile = (1.0 - papr_target_y) * 100.0
cm_percentile = (1.0 - cm_target_y) * 100.0

# 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
# (This completely replaces the need for y_axis, np.where, and manual sorting!)
# papr_vlines = [
#     np.percentile(unclipped_papr, papr_percentile),
#     np.percentile(icf_papr[-1], papr_percentile),
#     np.percentile(pred_papr, papr_percentile)
# ]

cm_vlines = [
    np.percentile(unclipped_cm, cm_percentile),
    np.percentile(icf_cm[-1], cm_percentile),
    np.percentile(pred_cm, cm_percentile)
]

# cm_vlines = [
#     np.max(unclipped_cm),
#     np.max(icf_cm),
#     np.max(pred_cm)
# ]

# plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)
# plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)
# labels = ["OG", "ICF 1", "ICF 2", "ICF 3"]
# plot_ccdf_compare([unclipped_papr, *icf_papr], label=labels, metric="papr")
# plot_ccdf_compare([unclipped_cm, *icf_cm], label=labels, metric="cm")

end = time.time()
# ~8s
print(f"Total execution time: {end - start:.2f} seconds")
print(f"{tech.upper()} execution time: {middle - start:.2f} seconds")
print(f"NN{tech.upper()} execution time: {end - middle:.2f} seconds")
print()
print("time taken in the inner iterations loop:", icf_time)
print("time taken in outer samples_per_L loop:", samples_per_L_minus_icf_time)

labels = ['Original', f'Clipped ({iterations} iteration{"s" if iterations > 1 else ""})', f'ICF ({iterations} iteration{"s" if iterations > 1 else ""})', f'NN{tech.upper()} Predicted']
# plot_signals(signals, labels)
plot_signals2(signals, labels)

signals_re = {}
signals_im = {}
signals_mag = {}
signals_ph = {}
for k, v in signals.items():
    signals_re[k] = v.real
    signals_im[k] = v.imag
    signals_mag[k] = np.abs(v)
    signals_ph[k] = np.angle(v)
labels2 = []
for l in labels:
    l += " (Real)"
    labels2.append(l)
plot_signals2(signals_re, labels2)
labels2 = []
for l in labels:
    l += " (Imaginary)"
    labels2.append(l)
plot_signals2(signals_im, labels2)
labels2 = []
for l in labels:
    l += " (Magnitude)"
    labels2.append(l)
plot_signals2(signals_mag, labels2)
labels2 = []
for l in labels:
    l += " (Phase)"
    labels2.append(l)
plot_signals2(signals_ph, labels2)
