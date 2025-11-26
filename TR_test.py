# import sys
import numpy as np
from matplotlib import pyplot as plt
# from scipy.special import erfc


# --- Helper Functions ---
def addCP(OFDM_time, CP):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])


def removeCP(signal, CP, N):
    return signal[CP:(CP + N)]


def gray_code(n):
    if n == 0: return [()]
    first_half = gray_code(n - 1)
    second_half = first_half[::-1]
    return [(0,) + code for code in first_half] + [(1,) + code for code in second_half]


def generate_64qam_mapping():
    gray_3bit = gray_code(3)
    levels = [-7, -5, -3, -1, 1, 3, 5, 7]
    mapping_table = {}
    for i_bits in gray_3bit:
        for q_bits in gray_3bit:
            bits = i_bits + q_bits
            i_val = levels[gray_3bit.index(i_bits)]
            q_val = levels[gray_3bit.index(q_bits)]
            mapping_table[bits] = complex(i_val, q_val)
    return mapping_table


def mapping(bits, modulation_type):
    modulated_signal = None
    demapping_table = None

    if modulation_type == 'QPSK':
        mapping_table = {(0, 0): 1 + 1j, (0, 1): 1 - 1j, (1, 1): -1 - 1j, (1, 0): -1 + 1j}
        modulated_signal = np.array([mapping_table[tuple(b)] for b in bits])
        norm_factor = np.sqrt(np.mean(np.abs(modulated_signal) ** 2))
        modulated_signal /= norm_factor
        demapping_table = {v / norm_factor: k for k, v in mapping_table.items()}

    elif modulation_type == '16QAM':
        # ... (omitted for brevity, same as before) ...
        mapping_table = {
            (0, 0, 0, 0): -3 - 3j, (0, 0, 0, 1): -3 - 1j, (0, 0, 1, 0): -3 + 3j, (0, 0, 1, 1): -3 + 1j,
            (0, 1, 0, 0): -1 - 3j, (0, 1, 0, 1): -1 - 1j, (0, 1, 1, 0): -1 + 3j, (0, 1, 1, 1): -1 + 1j,
            (1, 0, 0, 0): 3 - 3j, (1, 0, 0, 1): 3 - 1j, (1, 0, 1, 0): 3 + 3j, (1, 0, 1, 1): 3 + 1j,
            (1, 1, 0, 0): 1 - 3j, (1, 1, 0, 1): 1 - 1j, (1, 1, 1, 0): 1 + 3j, (1, 1, 1, 1): 1 + 1j
        }
        modulated_signal = np.array([mapping_table[tuple(b)] for b in bits])
        norm_factor = np.sqrt(np.mean(np.abs(modulated_signal) ** 2))
        modulated_signal /= norm_factor
        demapping_table = {v / norm_factor: k for k, v in mapping_table.items()}

    elif modulation_type == '64QAM':
        mapping_table = generate_64qam_mapping()
        modulated_signal = np.array([mapping_table[tuple(b)] for b in bits])
        norm_factor = np.sqrt(np.mean(np.abs(modulated_signal) ** 2))
        modulated_signal /= norm_factor
        demapping_table = {v / norm_factor: k for k, v in mapping_table.items()}

    return modulated_signal, demapping_table


# --- TONE RESERVATION FUNCTIONS ---


def tone_reservation(x_time, c_time, max_iter=10, clip_ratio=2.0, step=0.5):
    """
    Iteratively reduces PAPR by subtracting scaled/shifted versions of p_kernel.
    """
    x_tr = x_time.copy()

    # Target amplitude (clipping threshold)
    # We calculate this based on the average power of the current symbol
    avg_pwr = np.mean(np.abs(x_tr) ** 2)
    target_amp = np.sqrt(clip_ratio * avg_pwr)  # Target is sqrt(PAPR_target * P_avg)

    for _ in range(max_iter):
        # Find the maximum peak
        abs_x = np.abs(x_tr)
        max_val = np.max(abs_x)
        max_idx = np.argmax(abs_x)

        if max_val <= target_amp:
            break  # Target reached

        # Calculate the complex scaling factor alpha
        # We want to reduce the peak at max_idx down to target_amp
        # The correction vector is: alpha * p_shifted

        current_peak_complex = x_tr[max_idx]

        # How much to reduce? (Simple clipping approach)
        excess = max_val - target_amp

        # Phase of the peak
        phase = current_peak_complex / max_val

        # Scale factor: amount to remove * phase * step_size (mu)
        mu = step  # Convergence step size (0 < mu <= 1)
        alpha = mu * excess * phase

        # Create shifted kernel: p[n - max_idx]
        # Efficient circular shift
        p_shifted = np.roll(c_time, max_idx)

        # Subtract kernel from signal
        x_tr = x_tr - (alpha * p_shifted)

    return x_tr


# --- MAIN EXECUTION ---

if __name__ == "__main__":

    modulation_type = 'QPSK'
    N = 256 # number of subcarriers
    CP = N // 4 # cyclic prefix
    tr_ratio_test = 1/32 # percentage of subcarriers to be used as PRT
    target_papr_db = 6 # target PAPR
    K_PAPR = 5000 # number of symbols
    iterations = [5, 10, 15, 20, 30, 50] # number of iterations of Tone Reservation
    steps = 0.5 # step size
    L = 2 # oversampling factor
    N_oversampled = N * L

    if modulation_type == 'QPSK':
        mu = 2
    elif modulation_type == '16QAM':
        mu = 4
    elif modulation_type == '64QAM':
        mu = 6
    else:
        raise RuntimeError("Please enter valid modulation type")


    np.random.seed(42)  # Fix seed for reproducibility
    all_indices = np.arange(N)


    # print(f"Simulating PAPR with {K_PAPR} symbols...")
    # print(f"Reserved {num_reserved} tones for TR ({tr_ratio * 100:.1f}% overhead)")



    num_reserved = int(N * tr_ratio_test)
    reserved_indices = np.random.choice(all_indices, num_reserved, replace=False)
    data_indices = np.setdiff1d(all_indices, reserved_indices)

    # Generate PRTs
    C = np.zeros(N, dtype=complex)
    C[reserved_indices] = 1 + 0j
    c_time = np.fft.ifft(C)
    c_time = c_time / np.max(np.abs(c_time))

    # Calculate payload bits (only for data indices)
    payloadBits_per_signal = len(data_indices) * mu
    PAPR_original = []
    OFDM_freq_oversampled = np.zeros(N_oversampled, dtype=complex)
#! until here it run without issues
    # todo see if it is possible to merge both loops
    for i in range(K_PAPR):
        bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
        bits_SP = bits.reshape((len(data_indices), mu))

        mapped_symbols, _ = mapping(bits_SP, modulation_type)

        OFDM_freq = np.zeros(N, dtype=complex)
        OFDM_freq[data_indices] = mapped_symbols  # reserved_indices remain 0+0j
        OFDM_freq_oversampled[:N//2] = OFDM_freq[:N//2]
        OFDM_freq_oversampled[-N//2:] = OFDM_freq[N//2:]

        OFDM_time_oversampled = np.fft.ifft(OFDM_freq_oversampled)

        # calculating original PAPR before TR
        peak_pwr = np.max(np.abs(OFDM_time_oversampled) ** 2)
        avg_pwr = np.mean(np.abs(OFDM_time_oversampled) ** 2)
        PAPR_original.append(peak_pwr / avg_pwr)


    results = {}
    for iteration in iterations:
        PAPR_with_TR = []
        for i in range(K_PAPR):
            #if i % 100 == 0: print(f"Processing symbol {i}...")

            bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
            bits_SP = bits.reshape((len(data_indices), mu))

            mapped_symbols, _ = mapping(bits_SP, modulation_type)

            OFDM_freq = np.zeros(N, dtype=complex)
            OFDM_freq[data_indices] = mapped_symbols # reserved_indices remain 0+0j

            OFDM_time = np.fft.ifft(OFDM_freq)
            peak_val = np.max(np.abs(OFDM_time))
            avg_pwr = np.mean(np.abs(OFDM_time) ** 2)
            papr_db = 10 * np.log10((peak_val ** 2) / avg_pwr)
            # if papr_db > 9.0:
            #     target_ratio = 10 ** (target_papr_db / 10)
            #
            #     # 1. Run Tone Reservation
            #     OFDM_time_TR = tone_reservation(OFDM_time, c_time, iteration, target_ratio, steps)
            #
            #     # 2. Extract the Peak Cancelling Signal (c)
            #     # Since Output = Original + c, then c = Output - Original
            #     # (Note: Your TR function subtracts, so c will be negative, but magnitude is what matters)
            #     peak_cancelling_signal = OFDM_time_TR - OFDM_time
            #
            #     # Calculate new PAPR
            #     peak_val_tr = np.max(np.abs(OFDM_time_TR))
            #     avg_pwr_tr = np.mean(np.abs(OFDM_time_TR) ** 2)
            #     papr_tr_db = 10 * np.log10((peak_val_tr ** 2) / avg_pwr_tr)
            #
            #     print(f"Found Symbol #{i}")
            #     print(f"Original PAPR: {papr_db:.2f} dB")
            #     print(f"New PAPR:      {papr_tr_db:.2f} dB")
            #
            #     # --- PLOTTING ---
            #     plt.figure(figsize=(10, 8))
            #
            #     # Subplot 1: Original vs Output
            #     plt.subplot(2, 1, 1)
            #     plt.plot(np.abs(OFDM_time), 'b-', alpha=0.6, label='Original Signal')
            #     plt.plot(np.abs(OFDM_time_TR), 'r-', linewidth=1.5, label='Output Signal (with TR)')
            #
            #     # Draw Threshold Line
            #     target_amp = np.sqrt(target_ratio * avg_pwr)
            #     plt.axhline(y=target_amp, color='g', linestyle='--', label='Target Threshold')
            #
            #     plt.title(f'PAPR Reduction: {papr_db:.1f}dB -> {papr_tr_db:.1f}dB')
            #     plt.ylabel('Signal Magnitude')
            #     plt.legend(loc='upper right')
            #     plt.grid(True, alpha=0.3)
            #
            #     # Subplot 2: The "Noise" we added (Cancelling Signal)
            #     plt.subplot(2, 1, 2)
            #     plt.plot(peak_cancelling_signal, 'k-', label='Peak Cancelling Signal (c)')
            #     plt.title('Peak Cancelling Signal (Generated on Reserved Tones)')
            #     plt.xlabel('Time Samples')
            #     plt.ylabel('Magnitude')
            #     plt.legend(loc='upper right')
            #     plt.grid(True, alpha=0.3)
            #
            #     plt.tight_layout()
            #     plt.show()
            #
            #     break  # Stop after plotting one symbol

            target_ratio = 10 ** (target_papr_db / 10)
            OFDM_time_TR = tone_reservation(OFDM_time, c_time, iteration, target_ratio, steps)

            # calculating new PAPR after TR
            peak_pwr_tr = np.max(np.abs(OFDM_time_TR) ** 2)
            avg_pwr_tr = np.mean(np.abs(OFDM_time_TR) ** 2)
            PAPR_with_TR.append(peak_pwr_tr / avg_pwr_tr)


        results[iteration] = PAPR_with_TR


    PAPR_orig_dB = 10 * np.log10(PAPR_original)
    PAPR_tr_dB = 10 * np.log10(PAPR_with_TR)

    x_axis = np.linspace(4, 13, 200)

    ccdf_orig = [np.mean(PAPR_orig_dB > t) for t in x_axis]
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_axis, ccdf_orig, 'b-', lw=2, label='Original OFDM')

    colors = ['m', 'c', 'r', 'g', 'k', 'y']
    for idx, iteration in enumerate(iterations):
        papr_tr = results[iteration]
        papr_tr_db = 10 * np.log10(papr_tr)
        ccdf_tr = [np.mean(papr_tr_db > t) for t in x_axis]
        color = colors[idx]
        plt.semilogy(x_axis, ccdf_tr, '--', color=color, lw=2, label=f'TR OFDM with iterations={iteration:.0f}')

    plt.grid(True, which='both', alpha=0.5)
    plt.xlabel('PAPR0 (dB)')
    plt.ylabel('Pr(PAPR > PAPR0)')
    plt.title(f'CCDF Comparison: Different Alphas (N={N}, {modulation_type}) Reserved Tones={tr_ratio_test*N:.0f} Alpha={steps}')
    plt.ylim(1e-4, 1)
    plt.legend()
    #plt.show()

    target_papr_dbs = np.linspace(3, 11, 15)
    avg_power_increases = []

    K_POWER_SIM = 2000  # Use fewer symbols for power convergence to be fast
    fixed_step = 0.5  # Fix alpha for this comparison
    iterations = 50

    for target_db in target_papr_dbs:
        target_ratio_lin = 10 ** (target_db / 10)
        power_ratios = []

        for i in range(K_POWER_SIM):
            bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
            bits_SP = bits.reshape((len(data_indices), mu))
            mapped_symbols, _ = mapping(bits_SP, modulation_type)

            OFDM_freq = np.zeros(N, dtype=complex)
            OFDM_freq[data_indices] = mapped_symbols
            OFDM_time = np.fft.ifft(OFDM_freq)

            pwr_orig = np.mean(np.abs(OFDM_time) ** 2)

            # Apply TR
            OFDM_time_TR = tone_reservation(OFDM_time, c_time, iterations, target_ratio_lin, fixed_step)

            pwr_tr = np.mean(np.abs(OFDM_time_TR) ** 2)
            power_ratios.append(pwr_tr / pwr_orig)

        avg_increase_db = 10 * np.log10(np.mean(power_ratios))
        avg_power_increases.append(avg_increase_db)
        print(f"Target PAPR {target_db:.1f} dB -> Power Penalty {avg_increase_db:.2f} dB")

    # Plot 2: Power Penalty
    plt.figure(figsize=(10, 6))
    plt.plot(target_papr_dbs, avg_power_increases, 'bo-', linewidth=2)
    plt.title('Average Power Penalty vs. Target PAPR (Clipping Ratio)')
    plt.xlabel('Target PAPR (dB) [Clipping Ratio]')
    plt.ylabel('Average Power Increase (dB)')
    plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()
    plt.show()  # Show both plots now