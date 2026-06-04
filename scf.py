import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, scf, scf2, clip_and_filter_time
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.plots import plot_ccdf_compare, plot_ccdf, plot_signals, plot_signals2
import torch
from nnscf import NNSCFMapper, normalize, denormalize
import os
import time

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
iterations_str = f"{iterations} iteration{"s" if iterations > 1 else ""}"

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
if mod == "16qam":
    M = 16
elif mod == "qpsk":
    M = 4
modulate, _ = get_modem(M)

# Clipping technique
# tech = "icf"
tech = "scf"

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
# lr = 0.001
lr = 0.1
lr_list = str(float(lr)).split(".")
lr_str = f"dot{lr_list[1]}" if lr < 1 else f"{lr_list[0]}dot{lr_list[1]}"

# Optimizer
# opt = "Adam"
opt = "LBFGS"

# Data splitting (used only in saving and loading files, not in the actual C&F process)
data_size = 100
train_size = 70
val_size = 10
test_size = data_size - train_size - val_size
if test_size:
    one_batch = None
else:
    test_size = 1
    one_batch = "00"
train_val_test_str = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"

params = f"{opt} optimizer {tech.upper()} {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)\nlr = {lr} ({train_size} Training/{val_size} Validation/{test_size} Testing)"

relev = "./relevant files/"
os.makedirs(relev, exist_ok=True)

# * Load nnscf model

# 2. Load the trained models
# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_re = torch.load(
    os.path.join(relev, f"{opt}_{mod}_{tech}_{train_val_test_str}_{lr_str}_weights_limits_re.pth"),
    weights_only=False
)
checkpoint_im = torch.load(
    os.path.join(relev, f"{opt}_{mod}_{tech}_{train_val_test_str}_{lr_str}_weights_limits_im.pth"),
    weights_only=False
)

# Instantiate the empty models
NN_Mod_Re = NNSCFMapper().to(device)
NN_Mod_Im = NNSCFMapper().to(device)

# Inject the trained weights
NN_Mod_Re.load_state_dict(checkpoint_re['model_state_dict'])
NN_Mod_Im.load_state_dict(checkpoint_im['model_state_dict'])

NN_Mod_Re.eval()
NN_Mod_Im.eval()

# 2. Grab the saved normalization limits
X_r_min = checkpoint_re['X_min']
X_r_max = checkpoint_re['X_max']
Y_r_min = checkpoint_re['Y_min']
Y_r_max = checkpoint_re['Y_max']

X_i_min = checkpoint_im['X_min']
X_i_max = checkpoint_im['X_max']
Y_i_min = checkpoint_im['Y_min']
Y_i_max = checkpoint_im['Y_max']

# Simulation
# unclipped_papr, scf_papr, pred_papr, icf_papr = [], [], [], [[] for _ in range(iterations)]
unclipped_cm, scf_cm, pred_cm, icf_cm = [], [], [], [[] for _ in range(iterations)]

samples_per_L_minus_scf_icf_time = 0
scf_time = 0
icf_time = 0

x_time, x_clip_scf, x_scf, x_clip_icf, x_icf = [], [], [], [], []
rms = None

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
    samples_per_L_minus_scf_icf_time += t2 - t1

    # Process SCF (1 Step replacing the 3 ICF iterations)
    x_scf, x_clip_scf, rms = scf2(x_time, cr, N, iterations=iterations)

    # Store PAPR of the current iterative result
    # scf_papr.append(calculate_papr(x_scf))
    scf_cm.append(calculate_cm(x_scf))

    t3 = time.time()
    scf_time += t3 - t2

    # Process C&F
    x_icf = x_time.copy()
    for i in range(iterations):
        # ! RMS was already calculated during SCF
        x_icf, x_clip_icf, _ = clip_and_filter_time(x_icf, cr, N)

        # Store PAPR of the current iterative result
        # icf_papr[i].append(calculate_papr(x_icf))
        icf_cm[i].append(calculate_cm(x_icf))

    t4 = time.time()
    icf_time += t4 - t3

    # store the final iteration's real and imag parts separately
    rx_time[0].append(np.real(x_scf).astype(np.float32, copy=False))
    rx_time[1].append(np.imag(x_scf).astype(np.float32, copy=False))
    t5 = time.time()
    samples_per_L_minus_scf_icf_time += t5 - t4

tx_time = np.array(tx_time, dtype=np.float32)
rx_time = np.array(rx_time, dtype=np.float32)

signals = {
    'original': x_time,
    'clipped scf': x_clip_scf,
    'scf': x_scf,
    'clipped icf': x_clip_icf,
    'icf': x_icf
}
A = rms * cr

middle = time.time()

# * Compare with nnscf model

# 1. Normalize the data and turn it to torch tensors
tx_real = normalize(tx_time[0], (X_r_min, X_r_max))
tx_imag = normalize(tx_time[1], (X_i_min, X_i_max))

tx_real = torch.tensor(tx_time[0], dtype=torch.float32)
tx_imag = torch.tensor(tx_time[1], dtype=torch.float32)

# 3. Generate Predictions (No gradients needed for testing)
with torch.no_grad():
    # Memoryless flattening
    X_real_flat = tx_real.view(-1, 1).to(device)
    X_imag_flat = tx_imag.view(-1, 1).to(device)

    predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
    predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()


pred_denorm_real = denormalize(predicted_real, (Y_r_min, Y_r_max))
pred_denorm_imag = denormalize(predicted_imag, (Y_i_min, Y_i_max))


# 4. Reconstruct the Complex OFDM Signals
# each has shape of samples_per_L, (N * L)
predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

signals["predicted"] = predicted_complex[-1]

# 5. Calculate PAPR (Peak-to-Average Power Ratio) for CCDF
for i in range(samples_per_L):
    # pred_papr.append(calculate_papr(predicted_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))
# pred_papr = np.array(pred_papr)
pred_cm = np.array(pred_cm)

# 6. Plot the CCDF
title = f"NN{tech.upper()} Predicted OFDM\n{params}"
labels = ['Original', f'SCF ({iterations_str})', f'ICF ({iterations_str})', f'NN{tech.upper()} Predicted']
# papr_list = [unclipped_papr, scf_papr, icf_papr[-1], pred_papr]
cm_list = [unclipped_cm, scf_cm, icf_cm[-1], pred_cm]

# plot_ccdf_compare(papr_list, f"Original vs SCF vs ICF vs {title}", labels)
plot_ccdf_compare(cm_list, f"Original vs SCF vs ICF vs {title}", labels, metric="CM")

# fixme: both percentile and max are not the most efficient solutions

# todo: find a way to embed the floor part into the plotting function
# 1. Define the target y-levels (probabilities)
# papr_target_y = 1e-4
cm_target_y = 1e-3

# 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
# papr_percentile = (1.0 - papr_target_y) * 100.0
cm_percentile = (1.0 - cm_target_y) * 100.0

# 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
# (This completely replaces the need for y_axis, np.where, and manual sorting!)
# papr_vlines = [
#     np.percentile(unclipped_papr, papr_percentile),
#     np.percentile(scf_papr, papr_percentile),
#     np.percentile(icf_papr[-1], papr_percentile),
#     np.percentile(pred_papr, papr_percentile)
# ]

cm_vlines = [
    np.percentile(unclipped_cm, cm_percentile),
    np.percentile(scf_cm, cm_percentile),
    np.percentile(icf_cm[-1], cm_percentile),
    np.percentile(pred_cm, cm_percentile)
]

# cm_vlines = [
#     np.max(unclipped_cm),
#     np.max(icf_cm),
#     np.max(pred_cm)
# ]

# plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)
plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)
# labels = ["OG", "ICF 1", "ICF 2", "ICF 3"]
# plot_ccdf_compare([unclipped_papr, *icf_papr], label=labels, metric="papr")
# plot_ccdf_compare([unclipped_cm, *icf_cm], label=labels, metric="cm")

end = time.time()
# ~8s
print(f"Total execution time: {end - start:.2f} seconds")
print(f"{tech.upper()} execution time: {middle - start:.2f} seconds")
print(f"NN{tech.upper()} execution time: {end - middle:.2f} seconds")
print()
print("time taken in the scf calculations:", scf_time)
print("time taken in the icf calculations:", icf_time)
print("time taken in outer samples_per_L loop:", samples_per_L_minus_scf_icf_time)

labels = ['Original', 'Clipped (SCF)', f'SCF ({iterations_str})', f'Clipped ({iterations_str})', f'ICF ({iterations_str})', f'NN{tech.upper()} Predicted']
signals_image_path = os.path.join(relev, f"{opt}_{mod}_{tech}_{train_val_test_str}_{lr_str}_signals.png")
# plot_signals(signals, labels, A)
plot_signals2(signals, labels, A, signals_image_path)

signals_re = {}
signals_im = {}
for k, v in signals.items():
    signals_re[k] = v.real
    signals_im[k] = v.imag
labels2 = []
for l in labels:
    l += " (Real)"
    labels2.append(l)
plot_signals2(signals_re, labels2, A)
labels2 = []
for l in labels:
    l += " (Imaginary)"
    labels2.append(l)
plot_signals2(signals_im, labels2, A)
