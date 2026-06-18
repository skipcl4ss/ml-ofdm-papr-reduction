import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, scf, scf2, clip_and_filter_time, emulate_awgn_channel
from ofdm.metrics import ber_theoretical
from ofdm.plots import plot_ber, plot_constellation
import torch
from nnscf import NNSCFMapper, normalize, denormalize
import os
import time

# todo: clean up this and other scf files
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

# modulation scheme
# mod = "16qam"
mod = "qpsk"
# edits the stops to be same as the paper, and declares M for the modem
if mod == "16qam":
    M = 16
    stop = 14
elif mod == "qpsk":
    M = 4
    stop = 8
modulate, demodulate = get_modem(M)

# clipping technique
# tech = "icf"
tech = "scf"

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
lr = 0.001
# lr = 0.01
lr_str = "dot" + str(lr).split(".")[1]

# Data splitting (used only in saving and loading files, not in the actual C&F process)
datasize = 100
# datasize = 120
train_size = 80
# val_size = 20
val_size = 0
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
# NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_mod_re_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))
# NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_mod_im_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))
# NN_Mod_Re.load_state_dict(torch.load("./trained_models/Adam_16qam_scf_mod_re_weights_70_10_20_dot001.pth"))
# NN_Mod_Im.load_state_dict(torch.load("./trained_models/Adam_16qam_scf_mod_im_weights_70_10_20_dot001.pth"))
# NN_Mod_Re.load_state_dict(torch.load("./trained_models/Adam_qpsk_mod_re_weights_70_10_20_dot001.pth"))
# NN_Mod_Im.load_state_dict(torch.load("./trained_models/Adam_qpsk_mod_im_weights_70_10_20_dot001.pth"))
NN_Mod_Re.load_state_dict(torch.load("./test/Adam_qpsk_scf_mod_re_weights_70_10_20_dot001.pth"))
NN_Mod_Im.load_state_dict(torch.load("./test/Adam_qpsk_scf_mod_im_weights_70_10_20_dot001.pth"))

NN_Mod_Re.eval()
NN_Mod_Im.eval()

limits = torch.load(f"./relev {mod}/new_method_raw_limits.pt", weights_only=False)

def normalize(raw, limits):
    min, max = limits
    norm = 2.0 * ((raw - min) / (max - min)) - 1.0 if max != min else raw
    return norm

# ber_theory_list = []
BER_results = {
    'no clipping': [],
    f'clipped scf {iterations_str}': [],
    'scf': [],
    'predicted': []
}
for i in range(1, iterations + 1):
    BER_results[f'clipped icf ({f"{i} iteration{"s" if i > 1 else ""}"})'] = []
    BER_results[f'icf ({f"{i} iteration{"s" if i > 1 else ""}"})'] = []

# todo: make variable names more consistent

EbNo_minus_num_symb_loop = 0
num_symb_loop_minus_scf_icf_nn_time = 0
scf_time = 0
icf_time = 0
nn_time = 0

# todo: try to understand whether to downsample before awgn or not

EbNo_range = np.arange(0, stop + 1, 1) # does not affect ccdf
bits_per_symbol = int(np.log2(M))
num_symb = 100
for EbNo_dB in EbNo_range:
    t1 = time.time()

    plotted = False
    # ! the ratio at the end is to represent the loss done by the CP
    SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol * (N / (N + CP)))
    # Counters for bit errors
    bit_error_no_clip = 0
    bit_error_predicted = 0
    bit_error_clipped_scf = 0
    bit_error_scf = 0
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
            plot_constellation(tx_symbols, mod, limit, title=f'Transmitted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Tx Symbols', color='black')

        # No clipping case
        tx_ofdm_no_clip = np.fft.ifft(tx_symbols)

        # ? how does it differ
        rx_symbols_no_clip = emulate_awgn_channel(tx_ofdm_no_clip, CP, SNR_dB)
        # rx_symbols_no_clip = emulate_awgn_channel(tx_ofdm_no_clip, CP, SNR_dB, L)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(rx_symbols_no_clip, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (No Clip)', label='Rx Symbols (No Clip)', color='black')

        rx_bits_no_clip = demodulate(rx_symbols_no_clip, bits=True)

        bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

        t4 = time.time()
        num_symb_loop_minus_scf_icf_nn_time += t4 - t3

        # todo: maybe this loop could be cleaned up
        tx_time_oversampled = oversample_time(tx_symbols, N, L)
        # Process SCF (1 Step replacing the 3 ICF iterations)
        filtered_time_scf, clipped_time_scf, _ = scf2(tx_time_oversampled, cr, N, iterations=iterations)

        # rx_symbols_clipped_scf = emulate_awgn_channel(clipped_time_scf, CP, SNR_dB)
        rx_symbols_clipped_scf = emulate_awgn_channel(clipped_time_scf, CP, SNR_dB, L)

        rx_bits_clipped_scf = demodulate(rx_symbols_clipped_scf, bits=True)

        bit_error_clipped_scf += np.sum(tx_bits != rx_bits_clipped_scf)

        # rx_symbols_scf = emulate_awgn_channel(filtered_time_scf, CP, SNR_dB)
        rx_symbols_scf = emulate_awgn_channel(filtered_time_scf, CP, SNR_dB, L)

        rx_bits_scf = demodulate(rx_symbols_scf, bits=True)

        bit_error_scf += np.sum(tx_bits != rx_bits_scf)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            # plot_constellation(rx_symbols_clipped_scf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Clipped)', label='Rx Symbols (Clipped)', color='magenta')
            plot_constellation(rx_symbols_scf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='blue')

        t5 = time.time()
        scf_time += t5 - t4

        # Process each iteration
        filtered_time_icf = tx_time_oversampled.copy()
        rx_symbols_icf = None
        for i in range(1, iterations + 1):
            filtered_time_icf, clipped_time_icf, _ = clip_and_filter_time(filtered_time_icf, cr, N)

            # ? how does it differ
            # rx_symbols_clipped_icf = emulate_awgn_channel(clipped_time_icf, CP, SNR_dB)
            rx_symbols_clipped_icf = emulate_awgn_channel(clipped_time_icf, CP, SNR_dB, L)

            rx_bits_clipped_icf = demodulate(rx_symbols_clipped_icf, bits=True)

            bit_error_clipped_icf[i] += np.sum(tx_bits != rx_bits_clipped_icf)

            # ? how does it differ
            # rx_symbols_icf = emulate_awgn_channel(filtered_time_icf, CP, SNR_dB)
            rx_symbols_icf = emulate_awgn_channel(filtered_time_icf, CP, SNR_dB, L)

            rx_bits_icf = demodulate(rx_symbols_icf, bits=True)

            bit_error_icf[i] += np.sum(tx_bits != rx_bits_icf)

            # if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            #     plot_constellation(rx_symbols_clipped_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Clipped)', label='Rx Symbols (Clipped)', color='magenta')
            #     plot_constellation(rx_symbols_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='red')

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(rx_symbols_icf, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='red')

        t6 = time.time()
        icf_time += t6 - t5

        # * Compare with nnscf model

        # 1. Normalize the data and turn it to torch tensors
        # ? why tx_time_oversampled
        tx_real = normalize(tx_time_oversampled.real, limits['X_r_limits'])
        tx_imag = normalize(tx_time_oversampled.imag, limits['X_i_limits'])
        tx_real = torch.tensor(tx_real)
        tx_imag = torch.tensor(tx_imag)

        # 2. Generate Predictions
        with torch.no_grad():
            # BYPASS TEST: Comment out the network (until predicted_imag)
            # Memoryless flattening
            X_real_flat = tx_real.view(-1, 1).to(device)
            X_imag_flat = tx_imag.view(-1, 1).to(device)

            predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
            predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()

        pred_denorm_real = denormalize(predicted_real, limits['Y_r_limits'])
        pred_denorm_imag = denormalize(predicted_imag, limits['Y_i_limits'])

        predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

        # ? how does it differ
        # predicted_symbols = emulate_awgn_channel(predicted_complex, CP, SNR_dB)
        predicted_symbols = emulate_awgn_channel(predicted_complex, CP, SNR_dB, L)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(predicted_symbols, mod, limit, title=f'Predicted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Rx Symbols (Predicted)', color='green')
        plotted = True

        predicted_bits = demodulate(predicted_symbols, bits=True)

        bit_error_predicted += np.sum(tx_bits != predicted_bits)

        total_bits += len(tx_bits)
        t7 = time.time()
        nn_time += t7 - t6
    t8 = time.time()

    # Calculate BER for this Eb/No
    ber_no_clip = bit_error_no_clip / total_bits
    ber_theory = ber_theoretical(EbNo_dB, M)
    ber_clipped_scf = bit_error_clipped_scf / total_bits
    ber_scf = bit_error_scf / total_bits
    ber_predicted = bit_error_predicted / total_bits

    BER_results['no clipping'].append(ber_no_clip)
    BER_results['predicted'].append(ber_predicted)

    BER_results[f'clipped scf {iterations_str}'].append(ber_clipped_scf)
    BER_results[f'scf'].append(ber_scf)
    for i in range(1, iterations + 1):
        ber_clipped_icf = bit_error_clipped_icf[i] / total_bits
        ber_icf = bit_error_icf[i] / total_bits
        BER_results[f'clipped icf ({f"{i} iteration{"s" if i > 1 else ""}"})'].append(ber_clipped_icf)
        BER_results[f'icf ({f"{i} iteration{"s" if i > 1 else ""}"})'].append(ber_icf)
    # ber_theory_list.append(ber_theory)

    print(f"Eb/No: {EbNo_dB:.2f} dB | BER (No clip): {ber_no_clip:.6f} | BER (Theory): {ber_theory:.6f}")
    t9 = time.time()
    EbNo_minus_num_symb_loop += t9 - t8

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

labels = ["Unclipped", f"SCF ({iterations_str})", f"NN{tech.upper()} Predicted", f"ICF ({iterations_str})"]
plot_ber(EbNo_range, BER_list, title, labels, M)

end = time.time()
# ~s (num_symb = 1)
# ~3s (num_symb = 100)
print(f"Total execution time: {end - start:.2f} seconds")
print()
print("time taken in the scf calculations:", scf_time)
print("time taken in the icf calculations:", icf_time)
print(f"time taken in the NN{tech.upper()} calculations:", nn_time)
print("time taken in the middle num_symb loop:", num_symb_loop_minus_scf_icf_nn_time)
print("time taken in outer EbNo_range loop:", EbNo_minus_num_symb_loop)
