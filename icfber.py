import numpy as np
from ofdm.modem import qam16_mod, qpsk_mod, qam16_demod, qpsk_demod
from ofdm.candf import oversample_time, clip_and_filter_time, emulate_awgn_channel
from ofdm.metrics import ber_theoretical
from ofdm.plots import plot_ber, plot_constellation
import torch
from nnicf import NNICFMapper, normalize, denormalize
import os
import time

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

# modulation scheme
mod = "16qam"
M = 16
# mod = "qpsk"
# M = 4

# Hyperparameters (used only in saving and loading files, not in the actual C&F process)
lr = 0.001
# lr = 0.01
lr_str = "dot" + str(lr).split(".")[1]

# Data splitting (used only in saving and loading files, not in the actual C&F process)
train_size = 70
val_size = 10
test_size = 100 - train_size - val_size
if test_size:
    one_batch = None
else:
    test_size = 1
    one_batch = "00"
train_val_test_str = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"

# Optimizer
opt = "Adam"
# opt = "LBFGS"

params = f"{opt} optimizer {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)\nlr = {lr} ({train_size} Training/{val_size} Validation/{test_size} Testing)"

# model_dir = "./trained_models/"
model_dir = "./new architecture/"

# * Load nnicf model

# 2. Load the trained models
# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNICFMapper().to(device)
NN_Mod_Im = NNICFMapper().to(device)

# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_mod_re_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{opt}_{mod}_mod_im_weights_{train_val_test_str}_{lr_str}.pth"), weights_only=True))

NN_Mod_Re.eval()
NN_Mod_Im.eval()


ber_theory_list = []
BER_results = {
    'no_clipping': [],
    'predicted': []
}
for i in range(1, iterations + 1):
    BER_results[f'clipped_iteration {i}'] = []
    BER_results[f'clipped_filtered_iteration {i}'] = []

# todo: make variable names more consistent

EbNo_minus_num_symb_loop = 0
num_symb_minus_iterations_loop_minus_nn = 0
iterations_loop = 0
nn_calc_time = 0

# edits the stops to be same as the paper
if mod == "16qam":
    stop = 14
elif mod == "qpsk":
    stop = 8
EbNo_range = np.arange(0, stop + 1, 1) # does not affect ccdf
bits_per_symbol = int(np.log2(M))
num_symb = 100
for EbNo_dB in EbNo_range:
    t1 = time.time()

    plotted = False
    # ! the ratio at the en is to represent the loss done by the CP
    SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol * (N / (N + CP)))
    # Counters for bit errors
    bit_error_no_clip = 0
    bit_error_predicted = 0
    bit_error_clipped = {i: 0 for i in range(1, iterations + 1)}
    bit_error_filtered = {i: 0 for i in range(1, iterations + 1)}
    total_bits = 0

    t2 = time.time()
    EbNo_minus_num_symb_loop += t2 - t1
    for _ in range(num_symb):
    # for _ in range(1):
        t3 = time.time()
        # Generate random bits and modulate
        tx_bits = np.random.randint(0, 2, int(N * bits_per_symbol))

        if mod == "16qam":
            # Generate 16-QAM Symbols
            tx_symbols = qam16_mod(tx_bits, bits=True)
        elif mod == "qpsk":
            # Generate QPSK Symbols
            tx_symbols = qpsk_mod(tx_bits, bits=True)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            limit = np.max(np.abs(tx_symbols)) * 1.1
            plot_constellation(tx_symbols, mod, limit, title=f'Transmitted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Tx Symbols', color='black')

        # No clipping case
        tx_ofdm_no_clip = np.fft.ifft(tx_symbols)

        rx_symbols_no_clip = emulate_awgn_channel(tx_ofdm_no_clip, CP, SNR_dB)

        if mod == "16qam":
            rx_bits_no_clip = qam16_demod(rx_symbols_no_clip, bits=True)
        elif mod == "qpsk":
            rx_bits_no_clip = qpsk_demod(rx_symbols_no_clip, bits=True)

        bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

        # todo: maybe this loop could be cleaned up
        tx_time_oversampled_base = oversample_time(tx_symbols, N, L)
        # Process each iteration

        t4 = time.time()
        num_symb_minus_iterations_loop_minus_nn += t4 - t3

        rx_symbols_filtered = None
        for i in range(1, iterations + 1):
            # Get clipped signal (no filtering)
            tx_time_oversampled = oversample_time(tx_symbols, N, L)
            clipped_filtered_time, clipped_time = clip_and_filter_time(tx_time_oversampled, cr, N)
            # clipped_time = clip_time(tx_time_oversampled, cr)

            # # Downsample back to original rate (take every L-th sample)
            # clipped_downsampled = clipped_time[::L]

            rx_symbols_clipped = emulate_awgn_channel(clipped_time, CP, SNR_dB, L)

            if mod == "16qam":
                rx_bits_clipped = qam16_demod(rx_symbols_clipped, bits=True)
            elif mod == "qpsk":
                rx_bits_clipped = qpsk_demod(rx_symbols_clipped, bits=True)

            bit_error_clipped[i] += np.sum(tx_bits != rx_bits_clipped)


            # Get clipped + filtered signal
            # clipped_filtered_time, _ = clip_and_filter_time(tx_time_oversampled, cr, N)
            filtered_downsampled = clipped_filtered_time[::L]

            # rx_symbols_filtered = emulate_awgn_channel(filtered_downsampled, CP, SNR_dB)
            rx_symbols_filtered = emulate_awgn_channel(clipped_filtered_time, CP, SNR_dB, L)

            if mod == "16qam":
                rx_bits_filtered = qam16_demod(rx_symbols_filtered, bits=True)
            elif mod == "qpsk":
                rx_bits_filtered = qpsk_demod(rx_symbols_filtered, bits=True)

            bit_error_filtered[i] += np.sum(tx_bits != rx_bits_filtered)

            # if EbNo_dB == EbNo_range[0] and not plotted:
            #     plot_constellation(tx_symbols, mod, limit, title=f'Transmitted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Tx Symbols', color='black')
            #     plot_constellation(rx_symbols_no_clip, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (No Clip)', label='Rx Symbols (No Clip)', color='red')
            #     plot_constellation(rx_symbols_clipped, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Clipped)', label='Rx Symbols (Clipped)', color='green')
            #     plot_constellation(rx_symbols_filtered, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='blue')

            # ? why not rx_symbols_filtered
            tx_symbols = np.fft.fft(filtered_downsampled)
            # tx_symbols = rx_symbols_filtered

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(rx_symbols_filtered, mod, limit, title=f'Received Constellation at E$_b$/N$_0$ = {EbNo_dB}dB (Filtered)', label='Rx Symbols (Filtered)', color='blue')
        t5 = time.time()
        iterations_loop += t5 - t4

        # 1. Normalize the data and turn it to torch tensors
        # ? why tx_time_oversampled_base
        tx_real, minmax_real = normalize(tx_time_oversampled_base.real)
        tx_imag, minmax_imag = normalize(tx_time_oversampled_base.imag)

        # 2. Generate Predictions
        with torch.no_grad():
            # BYPASS TEST: Comment out the network (until predicted_imag)
            # Memoryless flattening
            X_real_flat = tx_real.view(-1, 1).to(device)
            X_imag_flat = tx_imag.view(-1, 1).to(device)

            predicted_real = NN_Mod_Re(X_real_flat).view_as(tx_real).cpu().numpy()
            predicted_imag = NN_Mod_Im(X_imag_flat).view_as(tx_imag).cpu().numpy()

            # # Pass the original input straight through
            # predicted_real = tx_real_tensor.squeeze(0).cpu().numpy().flatten()
            # predicted_imag = tx_imag_tensor.squeeze(0).cpu().numpy().flatten()
            # # plt.plot(tx_real_tensor.squeeze().cpu().numpy(), label="Original Input (Normalized)")
            # # plt.plot(predicted_real, label="NN Output (Flatlined)")
            # # plt.legend()
            # # plt.show()
            # # exit()

        pred_denorm_real = denormalize(predicted_real, minmax_real)
        pred_denorm_imag = denormalize(predicted_imag, minmax_imag)

        predicted_complex = pred_denorm_real + 1j * pred_denorm_imag

        # predicted_downsampled = predicted_complex[::L]
        predicted_symbols = emulate_awgn_channel(predicted_complex, CP, SNR_dB, L)

        if (EbNo_dB == EbNo_range[0] or EbNo_dB == EbNo_range[-1]) and not plotted:
            plot_constellation(predicted_symbols, mod, limit, title=f'Predicted Constellation at E$_b$/N$_0$ = {EbNo_dB}dB', label='Rx Symbols (Predicted)', color='magenta')
        plotted = True

        if mod == "16qam":
            predicted_bits = qam16_demod(predicted_symbols, bits=True)
        elif mod == "qpsk":
            predicted_bits = qpsk_demod(predicted_symbols, bits=True)

        bit_error_predicted += np.sum(tx_bits != predicted_bits)

        total_bits += len(tx_bits)
        t6 = time.time()
        nn_calc_time += t6 - t5
    t7 = time.time()

    # Calculate BER for this Eb/No
    ber_no_clip = bit_error_no_clip / total_bits
    ber_theory = ber_theoretical(EbNo_dB, M)
    ber_predicted = bit_error_predicted / total_bits

    BER_results['no_clipping'].append(ber_no_clip)
    BER_results['predicted'].append(ber_predicted)

    for i in range(1, iterations + 1):
        ber_clipped = bit_error_clipped[i] / total_bits
        ber_filtered = bit_error_filtered[i] / total_bits
        BER_results[f'clipped_iteration {i}'].append(ber_clipped)
        BER_results[f'clipped_filtered_iteration {i}'].append(ber_filtered)
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

labels = ["Unclipped", "ICF 3 Iterations", "Predicted"]
plot_ber(EbNo_range, [BER_results["no_clipping"], BER_results[f'clipped_filtered_iteration {iterations - 1}'], BER_results['predicted']], title, labels, M)

end = time.time()
# ~s (num_symb = 1)
# ~3s (num_symb = 100)
print(f"\nTotal execution time: {end - start:.2f} seconds")
print()
print("time taken in the inner iterations loop:", iterations_loop)
print("time taken in the nnicf calculations:", nn_calc_time)
print("time taken in the middle iterations loop:", num_symb_minus_iterations_loop_minus_nn)
print("time taken in outer EbNo loop loop:", EbNo_minus_num_symb_loop)
