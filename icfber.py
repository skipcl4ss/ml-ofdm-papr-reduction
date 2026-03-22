import numpy as np
from ofdm.modem import qam16_mod, qam16_demod
from ofdm.candf import clip_time, clip_and_filter_ofdm
from ofdm.awgn import awgn
from ofdm.ber import ber_theoretical
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
N = 256                 # Number of Subcarriers
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3
# ? revise the values
M = 16
# M = 2
CP = 32                 # Cyclic Prefix
# CP = N // 4

BER_results = {
    'no_clipping': [],
    'theoretical': []
}

for iter in range(1, iterations + 1):
    BER_results[f'clipped_iteration {iter}'] = []
    BER_results[f'clipped_filtered_iteration {iter}'] = []

EbNo_range = np.arange(0, 11, 1) # does not affect ccdf
bits_per_symbol = int(np.log2(M))
num_symb = 100

for EbNo_dB in EbNo_range:
    SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol)
    # SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol * (N / (N + CP)))
    # Counters for bit errors
    bit_error_no_clip = 0
    bit_error_clipped = {iter: 0 for iter in range(1, iterations + 1)}
    bit_error_filtered = {iter: 0 for iter in range(1, iterations + 1)}
    total_bits = 0

    for _ in range(num_symb):
        # Generate random bits and modulate
        tx_bits = np.random.randint(0, 2, int(N * bits_per_symbol))
        tx_symbols = qam16_mod(tx_bits)

        # --- No clipping case ---
        # tx_ofdm_no_clip = np.fft.ifft(tx_symbols, N)
        tx_ofdm_no_clip = np.fft.ifft(tx_symbols)
        tx_signal_no_clip = np.concatenate([tx_ofdm_no_clip[-CP:], tx_ofdm_no_clip])
        rx_signal_no_clip = awgn(tx_signal_no_clip, SNR_dB)
        rx_ofdm_no_clip = rx_signal_no_clip[CP:]
        # rx_symbols_no_clip = np.fft.fft(rx_ofdm_no_clip, N)
        rx_symbols_no_clip = np.fft.fft(rx_ofdm_no_clip)
        rx_bits_no_clip = qam16_demod(rx_symbols_no_clip)
        # M = 2 / M = 16 / M = 16 (no N in fft) / M = 16, CP = 64 (no N in fft)
        print("tx_bits", tx_bits.shape) # 256 / 1024
        print("tx_symbols", tx_symbols.shape) # 256 / 1024 / 1024
        print("tx_ofdm_no_clip", tx_ofdm_no_clip.shape) # 256 / 256 / 1024
        print(np.fft.ifft(tx_symbols).shape) # 256 / 1024 / 1024
        print("tx_signal_no_clip", tx_signal_no_clip.shape) # 288 (256 + 32 CP) / 288 (256 + 32 CP) / 1056 (1024 + 32 CP)
        print("rx_signal_no_clip", rx_signal_no_clip.shape) # 288 (256 + 32 CP) / 288 (256 + 32 CP) / 1056 (1024 + 32 CP)
        print("rx_ofdm_no_clip", rx_ofdm_no_clip.shape) # 256 / 256 / 1024
        print(np.fft.ifft(rx_ofdm_no_clip).shape) # 256 / 256 / 1024
        print("rx_symbols_no_clip", rx_symbols_no_clip.shape) # 256 / 256 / 1024
        print("rx_bits_no_clip", rx_bits_no_clip.shape) # 256 / 256 / 1024
        bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

        tx_current = tx_symbols
        # --- Process each iteration ---
        for iter in range(1, iterations + 1):
            # Get clipped signal (no filtering)
            mid = N // 2
            zeros = np.zeros((L - 1) * N, dtype=complex)
            oversampled_freq = np.concatenate([tx_current[:mid], zeros, tx_current[mid:]])
            tx_time_oversampled = np.fft.ifft(oversampled_freq)
            clipped_time = clip_time(tx_time_oversampled, cr)
            print("tx_current", tx_current.shape) # 256 / error / 1024
            print("zeros", zeros.shape) # 768 / error / 768
            print("oversampled_freq", oversampled_freq.shape) # 1024 / error / 1792
            print(np.fft.ifft(oversampled_freq, N).shape) # 256 / /
            print("tx_time_oversampled", tx_time_oversampled.shape) # 1024 / error / 1792
            print("clipped_time", clipped_time.shape) # 1024 / error / 1792

            # Downsample back to original rate (take every L-th sample)
            clipped_downsampled = clipped_time[::L]
            tx_signal_clipped = np.concatenate([clipped_downsampled[-CP:], clipped_downsampled])
            rx_signal_clipped = awgn(tx_signal_clipped, SNR_dB)
            rx_ofdm_clipped = rx_signal_clipped[CP:]
            rx_symbols_clipped = np.fft.fft(rx_ofdm_clipped, N)
            rx_bits_clipped = qam16_demod(rx_symbols_clipped)
            print("clipped_downsampled", clipped_downsampled.shape) # 256 / error / 448
            print("tx_signal_clipped", tx_signal_clipped.shape) # 288 (256 + 32 CP) / error / 480 (448 + 32 CP)
            print("rx_signal_clipped", rx_signal_clipped.shape) # 288 / error / 480
            print("rx_ofdm_clipped", rx_ofdm_clipped.shape) # 256 / error / 448
            print(np.fft.fft(rx_ofdm_clipped).shape) # 256 / /
            print("rx_symbols_clipped", rx_symbols_clipped.shape) # 256 / error / 256
            print("rx_bits_clipped", rx_bits_clipped.shape) # 256 / error / 256
            bit_error_clipped[iter] += np.sum(tx_bits != rx_bits_clipped)

            # Get clipped + filtered signal
            clipped_filtered_time = clip_and_filter_ofdm(tx_current, N, L=L, CR=cr)
            filtered_downsampled = clipped_filtered_time[::L]
            tx_signal_filtered = np.concatenate([filtered_downsampled[-CP:], filtered_downsampled])
            rx_signal_filtered = awgn(tx_signal_filtered, SNR_dB)
            rx_ofdm_filtered = rx_signal_filtered[CP:]
            rx_symbols_filtered = np.fft.fft(rx_ofdm_filtered, N)
            rx_bits_filtered = qam16_demod(rx_symbols_filtered)
            print("clipped_filtered_time", clipped_filtered_time.shape) # 1024
            print("filtered_downsampled", filtered_downsampled.shape) # 256
            print("tx_signal_filtered", tx_signal_filtered.shape) # 288 (256 + 32 CP)
            print("rx_signal_filtered", rx_signal_filtered.shape) # 288
            print("rx_ofdm_filtered", rx_ofdm_filtered.shape) # 256
            print(np.fft.fft(rx_ofdm_filtered).shape) # 256 / /
            print("rx_symbols_filtered", rx_symbols_filtered.shape) # 256
            print("rx_bits_filtered", rx_bits_filtered.shape) # 256
            bit_error_filtered[iter] += np.sum(tx_bits != rx_bits_filtered)

            # tx_current = clipped_filtered_time[::L]  # ? why not rx_symbols_filtered
            tx_current = rx_symbols_filtered

        total_bits += len(tx_bits)

    # Calculate BER for this Eb/No
    ber_no_clip = bit_error_no_clip / total_bits
    ber_theory = ber_theoretical(EbNo_dB, M)

    BER_results['no_clipping'].append(ber_no_clip)
    BER_results['theoretical'].append(ber_theory)

    for iter in range(1, iterations + 1):
        ber_clipped = bit_error_clipped[iter] / total_bits
        ber_filtered = bit_error_filtered[iter] / total_bits
        BER_results[f'clipped_iteration {iter}'].append(ber_clipped)
        BER_results[f'clipped_filtered_iteration {iter}'].append(ber_filtered)

    print(f"  Eb/No: {EbNo_dB:.2f} dB | BER (No clip): {ber_no_clip:.6f} | BER (Theory): {ber_theory:.6f}")

# Plot BER curves
plt.figure(figsize=(12, 8))

# Generate colors
colors_clipped = cm.plasma(np.linspace(0.1, 0.9, iterations))
colors_filtered = cm.viridis(np.linspace(0.1, 0.9, iterations))

# Plot no clipping (simulated)
plt.semilogy(EbNo_range, BER_results['no_clipping'], 'o-', color='gray', linewidth=2, markersize=8, label='No Clipping (Simulated)')

# Plot clipped curves
for iter in range(1, iterations + 1):
    plt.semilogy(EbNo_range, BER_results[f'clipped_iteration {iter}'], color=colors_clipped[iter - 1], linestyle='-', marker='*', linewidth=2, markersize=8, label=f'Clipped Iteration {iter}')

# Plot theoretical
plt.semilogy(EbNo_range, BER_results['theoretical'], 'k-', linewidth=2.5, label='Theoretical')

# Plot clipped + filtered curves
for iter in range(1, iterations + 1):
    plt.semilogy(EbNo_range, BER_results[f'clipped_filtered_iteration {iter}'], color=colors_filtered[iter - 1], linestyle=':', marker='o', linewidth=2, markersize=6, label=f'Clipped + Filtered Iteration {iter}')

plt.xlabel('SNR [dB]', fontsize=12)
plt.ylabel('BER', fontsize=12)
plt.title(f'BER vs SNR - 16QAM with Clipping (L={L})', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=9, loc='best', ncol=2)
plt.xlim([EbNo_range[0], EbNo_range[-1]])
# plt.ylim([1e-4, 1e0])
plt.tight_layout()
plt.show()
