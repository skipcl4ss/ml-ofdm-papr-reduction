import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, clip_and_filter_time, emulate_awgn_channel
from ofdm.metrics import ber_theoretical
from ofdm.plots import plot_ber, plot_constellation
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
CP = N // 4             # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3
iterations_str = f"{iterations} iteration{"s" if iterations > 1 else ""}"

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
# edits the stops to be same as the paper, and declares M for the modem
if mod == "16qam":
    M = 16
    stop = 14
elif mod == "qpsk":
    M = 4
    stop = 8
modulate, demodulate = get_modem(M)

# Clipping technique
# tech = "icf"
tech = "scf"

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
# lr = 0.001
lr = 0.05
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
train_val_test = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"
one_batch = None

if not test_size:
    one_batch = '00'

params = f"{opt} optimizer {tech.upper()} {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)\nlr = {lr} ({train_size} Training/{val_size} Validation/{test_size} Testing)"

relev_dir = "./relevant files/"

# * Load nnscf model

# 2. Load the trained models
# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNSCFMapper().to(device)
NN_Mod_Im = NNSCFMapper().to(device)

# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(os.path.join(relev_dir, f"{opt}_{mod}_{tech}_mod_re_weights_{train_val_test}_{lr_str}.pth"), weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(os.path.join(relev_dir, f"{opt}_{mod}_{tech}_mod_im_weights_{train_val_test}_{lr_str}.pth"), weights_only=True))

NN_Mod_Re.eval()
NN_Mod_Im.eval()

# ber_theory_list = []
BER_results = {
    'no clipping': [],
    'predicted': []
}
for i in range(1, iterations + 1):
    BER_results[f'clipped icf ({f"{i} iteration{"s" if i > 1 else ""}"})'] = []
    BER_results[f'icf ({f"{i} iteration{"s" if i > 1 else ""}"})'] = []

# todo: make variable names more consistent

EbNo_minus_num_symb_loop = 0
num_symb_minus_icf_nn_time = 0
icf_time = 0
nn_time = 0

constel_image_path = os.path.join(relev_dir, f"{opt}_{mod}_{tech}_{train_val_test}_{lr_str}_")

EbNo_range = np.arange(0, stop + 1, 1) # does not affect ccdf
bits_per_symbol = int(np.log2(M))
num_symb = 100

tx_ofdm_no_clip = None
rx_symbols_no_clip = None
clipped_time_icf = None
rx_symbols_clipped_icf = None
filtered_time_icf = None
rx_symbols_icf = None
predicted_complex = None
predicted_symbols = None

for EbNo_dB in EbNo_range:
    t1 = time.time()

    plotted = False
    # ! the ratio at the end is to represent the loss done by the CP
    SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol * (N / (N + CP)))
    # Counters for bit errors
    bit_error_no_clip = 0
    bit_error_predicted = 0
    bit_error_clipped_icf = {i: 0 for i in range(1, iterations + 1)}
    bit_error_icf = {i: 0 for i in range(1, iterations + 1)}
    total_bits = 0

    t2 = time.time()
    EbNo_minus_num_symb_loop += t2 - t1
    for _ in range(num_symb):
    # for _ in range(1):
        t3 = time.time()
        # Generate random bits and modulate
        tx_bits = np.random.randint(0, 2, int(N * bits_per_symbol))

        tx_symbols = modulate(tx_bits, bits=True)

        if EbNo_dB == EbNo_range[0] and not plotted:
            limit = np.max(np.abs(tx_symbols)) * 1.1
            plot_constellation(tx_symbols, mod, limit, title=f'Transmitted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Tx Symbols', color='black', save=(constel_image_path+"Tx.png"))

        # No clipping case
        tx_ofdm_no_clip = np.fft.ifft(tx_symbols)

        rx_symbols_no_clip = emulate_awgn_channel(tx_ofdm_no_clip, CP, SNR_dB)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(rx_symbols_no_clip, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (No Clip)', label='Rx Symbols (No Clip)', color='black', save=(constel_image_path+f"Rx_no_clip_{EbNo_dB}db.png"))

        rx_bits_no_clip = demodulate(rx_symbols_no_clip, bits=True)

        bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

        t4 = time.time()
        num_symb_minus_icf_nn_time += t4 - t3

        # todo: maybe this loop could be cleaned up
        tx_time_oversampled = oversample_time(tx_symbols, N, L)

        # Process each iteration
        filtered_time_icf = tx_time_oversampled.copy()
        rx_symbols_icf = None
        for i in range(1, iterations + 1):
            filtered_time_icf, clipped_time_icf, _ = clip_and_filter_time(filtered_time_icf, cr, N)

            rx_symbols_clipped_icf = emulate_awgn_channel(clipped_time_icf, CP, SNR_dB, L)

            rx_bits_clipped_icf = demodulate(rx_symbols_clipped_icf, bits=True)

            bit_error_clipped_icf[i] += np.sum(tx_bits != rx_bits_clipped_icf)

            rx_symbols_icf = emulate_awgn_channel(filtered_time_icf, CP, SNR_dB, L)

            rx_bits_icf = demodulate(rx_symbols_icf, bits=True)

            bit_error_icf[i] += np.sum(tx_bits != rx_bits_icf)

            # if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            #     plot_constellation(rx_symbols_clipped_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Clipped)', label='Rx Symbols (Clipped)', color='magenta', save=(constel_image_path+f"Rx_clipped_icf_{EbNo_dB}db.png"))
            #     plot_constellation(rx_symbols_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='red', save=(constel_image_path+f"Rx_icf_{EbNo_dB}db.png"))

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(rx_symbols_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='red', save=(constel_image_path+f"Rx_icf_{EbNo_dB}db.png"))

        t5 = time.time()
        icf_time += t5 - t4

        # * Compare with nnscf model

        # 1. Normalize the data and turn it to torch tensors
        # ? why tx_time_oversampled
        tx_real, minmax_real = normalize(tx_time_oversampled.real)
        tx_imag, minmax_imag = normalize(tx_time_oversampled.imag)

        # 2. Generate Predictions
        with torch.no_grad():
            # BYPASS TEST: Comment out the network (until predicted_imag)
            # Memoryless flattening
            X_real_flat = tx_real.view(-1, 1).to(device)
            X_imag_flat = tx_imag.view(-1, 1).to(device)

            predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
            predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()

        pred_denorm_real = denormalize(predicted_real, minmax_real)
        pred_denorm_imag = denormalize(predicted_imag, minmax_imag)

        predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

        predicted_symbols = emulate_awgn_channel(predicted_complex, CP, SNR_dB, L)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(predicted_symbols, mod, limit, title=f'Predicted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Rx Symbols (Predicted)', color='green', save=(constel_image_path+f"Rx_pred_{EbNo_dB}db.png"))
        plotted = True

        predicted_bits = demodulate(predicted_symbols, bits=True)

        bit_error_predicted += np.sum(tx_bits != predicted_bits)

        total_bits += len(tx_bits)
        t6 = time.time()
        nn_time += t6 - t5
    t7 = time.time()

    # Calculate BER for this Eb/No
    ber_no_clip = bit_error_no_clip / total_bits
    ber_theory = ber_theoretical(EbNo_dB, M)
    ber_predicted = bit_error_predicted / total_bits

    BER_results['no clipping'].append(ber_no_clip)
    BER_results['predicted'].append(ber_predicted)

    for i in range(1, iterations + 1):
        ber_clipped_icf = bit_error_clipped_icf[i] / total_bits
        ber_icf = bit_error_icf[i] / total_bits
        BER_results[f'clipped icf ({f"{i} iteration{"s" if i > 1 else ""}"})'].append(ber_clipped_icf)
        BER_results[f'icf ({f"{i} iteration{"s" if i > 1 else ""}"})'].append(ber_icf)
    # ber_theory_list.append(ber_theory)

    print(f"Eb/No: {EbNo_dB:.2f} dB | BER (No clip): {ber_no_clip:.6f} | BER (Theory): {ber_theory:.6f}")
    t8 = time.time()
    EbNo_minus_num_symb_loop += t8 - t7

# Plot BER curves
title = f"BER vs SNR\n{params}"

# print(list(BER_results.keys()))
# todo: correct the legend order
# labels = ["Unclipped", "Predicted"]
# for i in range(iterations):
#     labels.append(f'Clipped Iteration {i + 1}')
#     labels.append(f'Clipped + Filtered Iteration {i + 1}')
# print(labels)
# plot_ber(EbNo_range, BER_results.values(), title, labels, M)
# # labels = ["Unclipped", "Predicted"] + [f"Clipped Iteration {i}" for i in range(1, iterations + 1)] + [f"Clipped + Filtered Iteration {i}" for i in range(1, iterations + 1)]
# # print(labels)
# # plot_ber(EbNo_range, BER_results.values(), title, labels, M)

BER_list = []
for k, v in BER_results.items():
    if "clipped" in k:
        continue
    elif "1" in k:
        continue
    elif "2" in k:
        continue

    BER_list.append(v)
# BER_list.append(ber_theory_list)

labels = ["Unclipped", f"NN{tech.upper()} Predicted", f"ICF ({iterations_str})"]
plot_ber(EbNo_range, BER_list, title, labels, M)

end = time.time()
# ~s (num_symb = 1)
# ~s (num_symb = 100)
print(f"Total execution time: {end - start:.2f} seconds")
print()
print("time taken in the icf calculations:", icf_time)
print(f"time taken in the NN{tech.upper()} calculations:", nn_time)
print("time taken in the middle num_symb loop:", num_symb_minus_icf_nn_time)
print("time taken in outer EbNo_range loop:", EbNo_minus_num_symb_loop)

tx_ofdm_no_clip = np.array(tx_ofdm_no_clip)
rx_symbols_no_clip = np.array(rx_symbols_no_clip)
clipped_time_icf = np.array(clipped_time_icf)
filtered_time_icf = np.array(filtered_time_icf)
rx_symbols_clipped_icf = np.array(rx_symbols_clipped_icf)
rx_symbols_icf = np.array(rx_symbols_icf)
predicted_complex = np.array(predicted_complex)
predicted_symbols = np.array(predicted_symbols)

print(tx_ofdm_no_clip.shape, "tx_ofdm_no_clip")
print(rx_symbols_no_clip.shape, "rx_symbols_no_clip")
print(clipped_time_icf.shape, "clipped_time_icf")
print(rx_symbols_clipped_icf.shape, "rx_symbols_clipped_icf")
print(filtered_time_icf.shape, "filtered_time_icf")
print(rx_symbols_icf.shape, "rx_symbols_icf")
print(predicted_complex.shape, "predicted_complex")
print(predicted_symbols.shape, "predicted_symbols")
