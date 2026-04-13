import numpy as np
from ofdm.ccdf import plot_ber
from ofdm.modem import qam16_mod, qpsk_mod, qam16_demod, qpsk_demod
from ofdm.candf import clip_time, clip_and_filter_ofdm, oversample_time, emulate_awgn_channel
from ofdm.metrics import ber_theoretical
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nnicf import NNICFMapper, normalize, normalize2, denormalize
import torch
from scipy import signal
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
lr_str = "dot" + str(lr).split(".")[1]

# Data splitting (used only in saving and loading files, not in the actual C&F process)
train_size = 80
test_size = (100 - train_size) or 1
train_test_str = f"{train_size}_{test_size}"

# * Compare with nnicf model

# 2. Load the trained models
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNICFMapper(N_fft).to(device)
NN_Mod_Im = NNICFMapper(N_fft).to(device)
# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load(f"./trained_models/{mod}_mod_re_weights_{train_test_str}_{lr_str}.pth", weights_only=True))
NN_Mod_Im.load_state_dict(torch.load(f"./trained_models/{mod}_mod_im_weights_{train_test_str}_{lr_str}.pth", weights_only=True))

NN_Mod_Re.eval()
NN_Mod_Im.eval()

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

BER_results = {
    'no_clipping': [],
    'theoretical': [],
    # 'predicted': []
}

for i in range(1, iterations + 1):
    BER_results[f'clipped_iteration {i}'] = []
    BER_results[f'clipped_filtered_iteration {i}'] = []

EbNo_range = np.arange(0, 11, 1) # does not affect ccdf
bits_per_symbol = int(np.log2(M))
weights = 1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=int)
num_symb = 100

# done: see a way to clean up clip_time and clip_and_filter_ofdm, and maybe make a function for CP
# todo: implement nnicf ber, as it now just outputs traditional icf

for EbNo_dB in EbNo_range:
    # ? which is more accurate
    # SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol)
    SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol * (N / (N + CP)))
    # Counters for bit errors
    bit_error_no_clip = 0
    bit_error_predicted = 0
    bit_error_clipped = {i: 0 for i in range(1, iterations + 1)}
    bit_error_filtered = {i: 0 for i in range(1, iterations + 1)}
    total_bits = 0

    for _ in range(num_symb):
    # for _ in range(1):
        # Generate random bits and modulate
        # tx_data = np.random.randint(0, M, N)
        tx_bits = np.random.randint(0, 2, int(N * bits_per_symbol))
        tx_bits_grouped = tx_bits.reshape(-1, bits_per_symbol)

        # todo: make a function this part
        # tx_bits_decimal = tx_bits_grouped.dot(weights)
        # Reshape back into OFDM symbol groups (e.g., batches of 256 subcarriers)
        # tx_data = tx_bits_decimal
        # tx_data = tx_bits_decimal.reshape(-1, N)
        tx_data = tx_bits_grouped.dot(weights)

        if mod == "16qam":
            # Generate 16-QAM Symbols
            tx_symbols = qam16_mod(tx_data)
        elif mod == "qpsk":
            # Generate QPSK Symbols
            tx_symbols = qpsk_mod(tx_data)

        # --- No clipping case ---
        tx_ofdm_no_clip = np.fft.ifft(tx_symbols)

        # tx_signal_no_clip = np.concatenate([tx_ofdm_no_clip[-CP:], tx_ofdm_no_clip])
        # rx_signal_no_clip = awgn(tx_signal_no_clip, SNR_dB)
        # rx_ofdm_no_clip = rx_signal_no_clip[CP:]
        # rx_symbols_no_clip = np.fft.fft(rx_ofdm_no_clip)
        rx_symbols_no_clip = emulate_awgn_channel(tx_ofdm_no_clip, CP, SNR_dB)

        if mod == "16qam":
            rx_data_no_clip = qam16_demod(rx_symbols_no_clip)
        elif mod == "qpsk":
            rx_data_no_clip = qpsk_demod(rx_symbols_no_clip)

        rx_bits_no_clip = []
        for j in rx_data_no_clip:
            for k in weights:
                rx_bits_no_clip.append(j // k)
                if j // k: j -= k
        rx_bits_no_clip = np.array(rx_bits_no_clip)

        bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

        # --- Process each iteration ---
        for i in range(1, iterations + 1):
            # Get clipped signal (no filtering)
            # zeros = np.zeros((L - 1) * N, dtype=complex)
            # oversampled_freq = np.concatenate([tx_symbols[:mid], zeros, tx_symbols[mid:]])
            # # fixed: problem was fixed by multiplying by L
            # tx_time_oversampled = np.fft.ifft(oversampled_freq) * L
            tx_time_oversampled = oversample_time(tx_symbols, N, L)
            clipped_time = clip_time(tx_time_oversampled, cr)

            # Downsample back to original rate (take every L-th sample)
            clipped_downsampled = clipped_time[::L]

            # tx_signal_clipped = np.concatenate([clipped_downsampled[-CP:], clipped_downsampled])
            # rx_signal_clipped = awgn(tx_signal_clipped, SNR_dB)
            # rx_ofdm_clipped = rx_signal_clipped[CP:]
            # rx_symbols_clipped = np.fft.fft(rx_ofdm_clipped)
            rx_symbols_clipped = emulate_awgn_channel(clipped_downsampled, CP, SNR_dB)

            if mod == "16qam":
                rx_data_clipped = qam16_demod(rx_symbols_clipped)
            elif mod == "qpsk":
                rx_data_clipped = qpsk_demod(rx_symbols_clipped)

            rx_bits_clipped = []
            for j in rx_data_clipped:
                for k in weights:
                    rx_bits_clipped.append(j // k)
                    if j // k: j -= k
            rx_bits_clipped = np.array(rx_bits_clipped)

            bit_error_clipped[i] += np.sum(tx_bits != rx_bits_clipped)

            # ! lfilter() simply does not work, and filtfilt() plot is slightly worse than clipped
            # symb = clip_time(oversample_time(tx_symbols, N, L), cr)
            # # clipped_filtered_time = signal.lfilter(b, a, symb).astype(np.complex64)
            # # clipped_filtered_time = signal.filtfilt(b, a, symb)
            # filtered_downsampled = clipped_filtered_time[::L]

            # Get clipped + filtered signal
            clipped_filtered_time = clip_and_filter_ofdm(tx_symbols, N, L, cr)
            filtered_downsampled = clipped_filtered_time[::L]

            # tx_signal_filtered = np.concatenate([filtered_downsampled[-CP:], filtered_downsampled])
            # rx_signal_filtered = awgn(tx_signal_filtered, SNR_dB)
            # rx_ofdm_filtered = rx_signal_filtered[CP:]
            # rx_symbols_filtered = np.fft.fft(rx_ofdm_filtered)
            rx_symbols_filtered = emulate_awgn_channel(filtered_downsampled, CP, SNR_dB)

            if mod == "16qam":
                rx_data_filtered = qam16_demod(rx_symbols_filtered)
            elif mod == "qpsk":
                rx_data_filtered = qpsk_demod(rx_symbols_filtered)

            rx_bits_filtered = []
            for j in rx_data_filtered:
                for k in weights:
                    rx_bits_filtered.append(j // k)
                    if j // k: j -= k
            rx_bits_filtered = np.array(rx_bits_filtered)

            # # fixme: cannot implement nnicf
            # # * nnicf part
            # if i < iterations - 1:
            #     tx_symbols = np.fft.fft(filtered_downsampled, N)  # ? why not rx_symbols_filtered
            #     # tx_symbols = rx_symbols_filtered
            #     # tx_symbols = np.fft.fft(filtered_downsampled)
            # else:
            #     # 1. Normalize the data and turn it to torch tensors
            #     tx_real = normalize2(tx_symbols)
            #     tx_test = denormalize(tx_real)
            #     tx_real, rx_real = normalize(tx_symbols, rx_symbols_filtered)
            #     tx_imag, rx_imag = normalize(tx_symbols, rx_symbols_filtered)
            #
            #     # 3. Generate Predictions (No gradients needed for testing)
            #     with torch.no_grad():
            #         predicted_real = NN_Mod_Re(tx_real).numpy()
            #         # predicted_imag = NN_Mod_Im(tx_imag).numpy()
            #
            #     pred_denorm_real = denormalize(predicted_real)
            #     pred_denorm_imag = denormalize(predicted_imag)
            #
            #     predicted_complex = pred_denorm_real + 1j * pred_denorm_imag
            #
            #     if mod == "16qam":
            #         predicted_data = qam16_demod(predicted_complex)
            #     elif mod == "qpsk":
            #         predicted_data = qpsk_demod(predicted_complex)
            #
            #     predicted_bits = []
            #     for j in predicted_data:
            #         for k in weights:
            #             predicted_bits.append(j // k)
            #             if j // k: j -= k
            #     predicted_bits = np.array(predicted_bits)
            #
            #     bit_error_predicted += np.sum(tx_bits != predicted_bits)

            bit_error_filtered[i] += np.sum(tx_bits != rx_bits_filtered)

            # tx_symbols = np.fft.fft(filtered_downsampled, N) # ? why not rx_symbols_filtered
            # # tx_symbols = rx_symbols_filtered
            # # tx_symbols = np.fft.fft(filtered_downsampled)

        total_bits += len(tx_bits)

    # Calculate BER for this Eb/No
    ber_no_clip = bit_error_no_clip / total_bits
    ber_theory = ber_theoretical(EbNo_dB, M)
    ber_predicted = bit_error_predicted / total_bits

    BER_results['no_clipping'].append(ber_no_clip)
    BER_results['theoretical'].append(ber_theory)
    # BER_results['predicted'].append(ber_predicted)

    for i in range(1, iterations + 1):
        ber_clipped = bit_error_clipped[i] / total_bits
        ber_filtered = bit_error_filtered[i] / total_bits
        BER_results[f'clipped_iteration {i}'].append(ber_clipped)
        BER_results[f'clipped_filtered_iteration {i}'].append(ber_filtered)

    print(f"  Eb/No: {EbNo_dB:.2f} dB | BER (No clip): {ber_no_clip:.6f} | BER (Theory): {ber_theory:.6f}")

# # Plot BER curves
# plt.figure(figsize=(12, 8))
#
# # Generate colors
# colors_clipped = cm.plasma(np.linspace(0.1, 0.9, iterations))
# colors_filtered = cm.viridis(np.linspace(0.1, 0.9, iterations))
#
# # Plot theoretical
# plt.semilogy(EbNo_range, BER_results['theoretical'], '--', color='black', linewidth=2.5, label='Theoretical')
# # Plot no clipping (simulated)
# plt.semilogy(EbNo_range, BER_results['no_clipping'], 'o-', color='gray', linewidth=2, markersize=8, label='No Clipping (Simulated)')
#
# # Plot clipped curves
# for i in range(1, iterations + 1):
#     plt.semilogy(EbNo_range, BER_results[f'clipped_iteration {i}'], color=colors_clipped[i - 1], linestyle='-', marker='*', linewidth=2, markersize=8, label=f'Clipped Iteration {i}')
# # Plot clipped + filtered curves
# for i in range(1, iterations + 1):
#     plt.semilogy(EbNo_range, BER_results[f'clipped_filtered_iteration {i}'], color=colors_filtered[i - 1], linestyle=':', marker='o', linewidth=2, markersize=6, label=f'Clipped + Filtered Iteration {i}')
#
# # plt.semilogy(EbNo_range, BER_results[f'clipped_filtered_iteration {iterations - 1}'], color=colors_filtered[iterations - 1], linestyle=':', marker='o', linewidth=2, markersize=6, label=f'Clipped + Filtered')
#
# # done: make a plotting function for this part
# plt.xlabel('SNR [dB]', fontsize=12)
# plt.ylabel('BER', fontsize=12)
# plt.title(f'BER vs SNR - 16QAM with Clipping (L={L})', fontsize=14)
# plt.grid(True, which='both', linestyle='--')
# plt.legend(fontsize=9, loc='best', ncol=2)
# plt.xlim([EbNo_range[0], EbNo_range[-1]])
# if mod == "16qam":
#     plt.ylim([1e-3, 1e0])
# elif mod == "qpsk":
#     plt.ylim([1e-4, 1e0])
# plt.tight_layout()
# plt.show()

# todo: needs more work in the color part
title = f"BER with {mod} and {iterations} iteration{"s" if i > 1 else ""}"
print(len(BER_results))
plot_ber(EbNo_range, BER_results.values(), title)

end = time.time()
# ~2s (num_symb = 1)
# ~33s (num_symb = 100)
print(f"\nTotal execution time: {end - start:.2f} seconds")
