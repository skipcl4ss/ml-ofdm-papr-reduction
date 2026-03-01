# ofdm_with_clipping_filtering.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.cm as cm
import time

start = time.time()

# ----------------- Modems (PSK and QAM) -----------------
class PSKModem:
    def __init__(self, M, phase_offset=None):
        self.M = M
        self.m = int(np.log2(M))

        if phase_offset is None:
            if M == 4:
                self.phase_offset = np.pi / 4
            else:
                self.phase_offset = 0
        else:
            self.phase_offset = phase_offset

        self.gray_map = self._generate_gray_map(self.m)
        self.bits_to_symbol = {}
        self.index_to_bits = {}

        for i, bit_str in enumerate(self.gray_map):
            phase = (2 * np.pi * i / self.M) + self.phase_offset
            symbol = np.cos(phase) + 1j * np.sin(phase)

            self.bits_to_symbol[bit_str] = symbol
            self.index_to_bits[i] = np.array([int(b) for b in bit_str])

    def _generate_gray_map(self, n):
        if n == 1:
            return ['0', '1']
        prev = self._generate_gray_map(n - 1)
        return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

    def modulate(self, bits):
        if len(bits) % self.m != 0:
            raise ValueError("Bitstream length must be a multiple of bits_per_symbol.")

        num_symbols = len(bits) // self.m
        bits_reshaped = bits.reshape(num_symbols, self.m)

        symbols = np.zeros(num_symbols, dtype=complex)
        for i in range(num_symbols):
            bit_str = "".join(str(b) for b in bits_reshaped[i])
            symbols[i] = self.bits_to_symbol[bit_str]

        return symbols

    def demodulate(self, symbols, mode='hard'):
        phases = np.angle(symbols) - self.phase_offset
        phases = np.mod(phases, 2 * np.pi)
        decisions = np.round(phases * self.M / (2 * np.pi)).astype(int) % self.M

        num_symbols = len(symbols)
        all_bits = np.zeros((num_symbols, self.m), dtype=int)

        for i in range(num_symbols):
            idx = decisions[i]
            all_bits[i] = self.index_to_bits[idx]

        return all_bits.flatten()


class QAMModem:
    def __init__(self, M):
        if int(np.sqrt(M))**2 != M:
            raise ValueError("M must be a square number (4, 16, 64, ...)")

        self.M = M
        self.m = int(np.log2(M))
        self.m_half = self.m // 2
        self.sqrt_M = int(np.sqrt(M))

        self.levels = np.arange(-self.sqrt_M + 1, self.sqrt_M, 2)
        gray_codes = self._generate_gray_map(self.m_half)

        self.bits_to_level = {b: l for b, l in zip(gray_codes, self.levels)}
        self.level_to_bits = {l: [int(x) for x in b] for b, l in zip(gray_codes, self.levels)}

    def _generate_gray_map(self, n):
        if n == 1:
            return ['0', '1']
        prev = self._generate_gray_map(n - 1)
        return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

    def modulate(self, bits):
        if len(bits) % self.m != 0:
            raise ValueError("Bitstream length must be a multiple of bits_per_symbol.")

        num_symbols = len(bits) // self.m
        bits_reshaped = bits.reshape(num_symbols, self.m)

        symbols = np.zeros(num_symbols, dtype=complex)

        for i in range(num_symbols):
            b_i = "".join(str(x) for x in bits_reshaped[i, :self.m_half])
            b_q = "".join(str(x) for x in bits_reshaped[i, self.m_half:])
            symbols[i] = self.bits_to_level[b_i] + 1j * self.bits_to_level[b_q]

        return symbols

    def demodulate(self, symbols, mode='hard'):
        min_l, max_l = self.levels[0], self.levels[-1]

        I_raw = np.clip(symbols.real, min_l, max_l)
        Q_raw = np.clip(symbols.imag, min_l, max_l)
        I_dec = 2 * np.round((I_raw + 1) / 2.0) - 1
        Q_dec = 2 * np.round((Q_raw + 1) / 2.0) - 1

        num_symbols = len(symbols)
        all_bits = np.zeros((num_symbols, self.m), dtype=int)

        for i in range(num_symbols):
            bits_i = self.level_to_bits[int(I_dec[i])]
            bits_q = self.level_to_bits[int(Q_dec[i])]
            all_bits[i] = bits_i + bits_q

        return all_bits.flatten()

# ----------------- Channel and theory -----------------
# ! takes a transmitted signal, applies noise, and outputs a recieved signal
def AWGN(tx_signal, SNR_dB):
    signal_power = np.mean(np.abs(tx_signal)**2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    return tx_signal + noise

# ! calculates theoretical BER at each Eb/No given the modulation type
def TBER(EbNo_dB, M):
    EbNo_lin = 10 ** (EbNo_dB / 10)

    if M == 2:
        return 0.5 * erfc(np.sqrt(EbNo_lin))
    elif M == 4:
        return 0.5 * erfc(np.sqrt(EbNo_lin))
    elif M == 8:
        k = 3
        return (1/k) * erfc(np.sqrt(k * EbNo_lin) * np.sin(np.pi/M))
    elif M == 16:
        return (3/8) * erfc(np.sqrt(0.4 * EbNo_lin))
    elif M == 64:
        return (7/24) * erfc(np.sqrt((1/7) * EbNo_lin))
    return 0.0

# ----------------- Utility functions -----------------
# ! calculates papr
def PAPR_from_time(tx_time):
    power = np.abs(tx_time)**2
    peak = np.max(power)
    avg = np.mean(power)
    if avg == 0: return 0.0
    return 10 * np.log10(peak / avg)

def theoretical_CCDF(N, papr_db_range):
    gamma = 10 ** (papr_db_range / 10)
    ccdf = 1 - (1 - np.exp(-gamma))**N
    return ccdf

def plot_CCDF_compare(papr_dict, N, title_suffix=''):
    plt.figure(figsize=(10, 7))
    colors = cm.viridis(np.linspace(0, 0.9, len(papr_dict)))
    for idx, (label, data) in enumerate(papr_dict.items()):
        if data.size == 0:
            continue
        sorted_papr = np.sort(data)
        y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr)
        # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
        if 'Clipped + Filtered' in label:
            linestyle = '--'
        elif 'Clipped' in label:
            linestyle = '-'
        else:
            linestyle = '-'
        plt.semilogy(sorted_papr, y_axis, color=colors[idx], linewidth=2, linestyle=linestyle, label=label)

    all_data = np.concatenate([v for v in papr_dict.values() if v.size > 0]) if papr_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_CCDF(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plt.ylim((10 ** -6, 10 ** 0))
    plt.xlabel('PAPR Threshold [dB]')
    plt.ylabel('Pr(PAPR > PAPR0)')
    plt.title(f'PAPR CCDF Comparison {title_suffix} (N={N})')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
# ----------------- Clipping & Filtering -----------------
def soft_clip_time(tx_time, CR):
    """
    Soft clipping in time domain:
      A = CR * RMS(tx_time)
      y(n) = x(n) if |x| <= A
             A * exp(j*angle(x)) if |x| > A
    """
    rms = np.sqrt(np.mean(np.abs(tx_time)**2))
    A = CR * rms
    mag = np.abs(tx_time)
    phase = np.angle(tx_time)
    clipped = np.where(mag <= A, tx_time, A * np.exp(1j * phase)) # ? why
    return clipped

def clip_and_filter_ofdm(freq_symbols, N, L=4, CR=2.23):
    """
    Steps:
      1) Create oversampled frequency vector by inserting zeros (L-1)*N in middle
      2) IFFT -> oversampled time
      3) Soft clip in time domain using CR
      4) FFT the clipped time -> keep only original subcarrier bins (i.e., set the inserted zeros bins to zero)
      5) IFFT back to oversampled time (filtered signal)
    Returns:
      tuple(original_oversampled_time, clipped_time_no_filter, clipped_filtered_time)
    """
    # build oversampled frequency vector (zero padding in middle)
    mid = N // 2
    zeros = np.zeros((L - 1) * N, dtype=complex)
    oversampled_freq = np.concatenate([freq_symbols[:mid], zeros, freq_symbols[mid:]])
    # time domain oversampled signal (original)
    tx_time_oversampled = np.fft.ifft(oversampled_freq)
    # soft clipping (no filtering)
    clipped_time = soft_clip_time(tx_time_oversampled, CR)
    # freq domain of clipped signal
    clipped_freq = np.fft.fft(clipped_time)
    # zero out the inserted bins (i.e., perform low-pass / band-limiting)
    # The original information lives in the positions we used earlier:
    kept_freq = np.zeros_like(clipped_freq)
    kept_freq[:mid] = clipped_freq[:mid]
    kept_freq[-mid:] = clipped_freq[-mid:]
    # kept_freq[-(N-mid):] = clipped_freq[-(N-mid):]
    # IFFT to get clipped+filtered time signal
    clipped_filtered_time = np.fft.ifft(kept_freq)

    return clipped_filtered_time
    # return tx_time_oversampled, clipped_time, clipped_filtered_time

# ----------------- Parameter input (kept interactive like original) -----------------
# # takes input from users
def param():
    print("Enter the OFDM Parameters")
    # This block asks the user for as integer as the number of subcarriers, then calculates the cyclic prefix from it
    while True:
        try:
            subc = int(input("Enter the number of subcarriers (Default: 512): ") or "512")
            cp = subc // 4
            break
        except ValueError:
            print("Kindly enter a valid number.")
    # This block asks for the modulation type and calculates the bits per symbol accordingly
    while True:
        mod_str = input("Enter the modulation type [BPSK, QPSK, 8PSK, 16QAM, 64QAM] (Default: 16QAM): ") or "16QAM"
        mod_dic = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64}
        if mod_str.upper() in mod_dic:
            M = mod_dic[mod_str.upper()]
            bits_per_symbol = int(np.log2(M))
            break
        else:
            print("Kindly enter a valid modulation type.")
    # This block asls for EbNo start and end as integers, and its step as a float
    while True:
        try:
            EbNo_start = int(input("Enter the start Eb/No (Default: 0): ") or "0")
            EbNo_end = int(input("Enter the stop Eb/No (Default: 10): " )or "10")
            EbNo_step = float(input("Enter the Eb/No step: (Default: 1): ") or "1")
            break
        except ValueError:
            print("Kindly enter a valid number.")
    # This block asks for the number of symbols as an integer
    while True:
        try:
            num_symb = int(input("Enter number of OFDM symbols (Default: 1000): ") or "1000")
            break
        except ValueError:
            print("Kindly enter a valid number")
    # Save all the results in a dictionary and return it as the function output
    parameters = {
        "subc": subc,
        "cp": cp,
        "mod_str": mod_str.upper(),
        "M": M,
        "bits_per_symbol": bits_per_symbol,
        "EbNo_range": np.arange(EbNo_start, EbNo_end+EbNo_step, EbNo_step),
        "num_symb": num_symb
    }
    return parameters

# ----------------- Main simulation -----------------
def main():
    # p = param()
    # print(p)
    p = {
        "subc": 128,
        "cp": 128 // 4, # subc // 4
        "mod_str": "QPSK",
        "M": 4,
        "bits_per_symbol": 2,
        # ! changeable
        "EbNo_range": np.arange(0, 11, 1), # does not affect ccdf
        "num_symb": int(input("Enter number of OFDM symbols (Default: 100): ") or "100")
    }
    subc = p["subc"]
    cp = p["cp"]
    M = p["M"]
    bits_per_symbol = p["bits_per_symbol"]
    num_symb = p["num_symb"]

    # choose modem
    if p["M"] in [2, 4, 8]:
        modem = PSKModem(M)
    else:
        modem = QAMModem(M)

    CR_list = [0.8, 1.0, 1.2, 1.4, 1.6]
    # ! useless
    L = 8  # Oversampling factor

    BER_results = {
        'no_clipping': [],
        'theoretical': []
    }

    for CR in CR_list:
        BER_results[f'clipped_CR={CR}'] = []
        BER_results[f'clipped_filtered_CR={CR}'] = []

    print(f"\nStarting BER Simulation for {p['mod_str']} with multiple CR values...")

    for EbNo_dB in p["EbNo_range"]:
        SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol)

        # Counters for bit errors
        bit_error_no_clip = 0
        bit_error_clipped = {CR: 0 for CR in CR_list}
        bit_error_filtered = {CR: 0 for CR in CR_list}
        total_bits = 0

        for _ in range(num_symb):
            # Generate random bits and modulate
            tx_bits = np.random.randint(0, 2, int(subc * bits_per_symbol))
            tx_symbols = modem.modulate(tx_bits)

            # --- No clipping case ---
            tx_ofdm_no_clip = np.fft.ifft(tx_symbols, subc)
            tx_signal_no_clip = np.concatenate([tx_ofdm_no_clip[-cp:], tx_ofdm_no_clip])
            rx_signal_no_clip = AWGN(tx_signal_no_clip, SNR_dB)
            rx_ofdm_no_clip = rx_signal_no_clip[cp:]
            rx_symbols_no_clip = np.fft.fft(rx_ofdm_no_clip, subc)
            rx_bits_no_clip = modem.demodulate(rx_symbols_no_clip, 'hard')
            bit_error_no_clip += np.sum(tx_bits != rx_bits_no_clip)

            # --- Process each CR value ---
            for CR in CR_list:
                # Get clipped signal (no filtering)
                mid = subc // 2
                zeros = np.zeros((L - 1) * subc, dtype=complex)
                oversampled_freq = np.concatenate([tx_symbols[:mid], zeros, tx_symbols[mid:]])
                tx_time_oversampled = np.fft.ifft(oversampled_freq)
                clipped_time = soft_clip_time(tx_time_oversampled, CR)

                # Downsample back to original rate (take every L-th sample)
                clipped_downsampled = clipped_time[::L]
                tx_signal_clipped = np.concatenate([clipped_downsampled[-cp:], clipped_downsampled])
                rx_signal_clipped = AWGN(tx_signal_clipped, SNR_dB)
                rx_ofdm_clipped = rx_signal_clipped[cp:]
                rx_symbols_clipped = np.fft.fft(rx_ofdm_clipped, subc)
                rx_bits_clipped = modem.demodulate(rx_symbols_clipped, 'hard')
                bit_error_clipped[CR] += np.sum(tx_bits != rx_bits_clipped)

                # Get clipped+filtered signal
                clipped_filtered_time = clip_and_filter_ofdm(tx_symbols, subc, L=L, CR=CR)
                filtered_downsampled = clipped_filtered_time[::L]
                tx_signal_filtered = np.concatenate([filtered_downsampled[-cp:], filtered_downsampled])
                rx_signal_filtered = AWGN(tx_signal_filtered, SNR_dB)
                rx_ofdm_filtered = rx_signal_filtered[cp:]
                rx_symbols_filtered = np.fft.fft(rx_ofdm_filtered, subc)
                rx_bits_filtered = modem.demodulate(rx_symbols_filtered, 'hard')
                bit_error_filtered[CR] += np.sum(tx_bits != rx_bits_filtered)

            total_bits += len(tx_bits)

        # Calculate BER for this Eb/No
        ber_no_clip = bit_error_no_clip / total_bits
        ber_theory = TBER(EbNo_dB, M)

        BER_results['no_clipping'].append(ber_no_clip)
        BER_results['theoretical'].append(ber_theory)

        for CR in CR_list:
            ber_clipped = bit_error_clipped[CR] / total_bits
            ber_filtered = bit_error_filtered[CR] / total_bits
            BER_results[f'clipped_CR={CR}'].append(ber_clipped)
            BER_results[f'clipped_filtered_CR={CR}'].append(ber_filtered)

        print(f"  Eb/No: {EbNo_dB:.2f} dB | BER (No clip): {ber_no_clip:.6f} | BER (Theory): {ber_theory:.6f}")

    # Plot BER curves
    plt.figure(figsize=(12, 8))

    # Generate colors
    colors_clipped = cm.plasma(np.linspace(0.1, 0.9, len(CR_list)))
    colors_filtered = cm.viridis(np.linspace(0.1, 0.9, len(CR_list)))

    # Plot clipped curves
    for idx, CR in enumerate(CR_list):
        plt.semilogy(p["EbNo_range"], BER_results[f'clipped_CR={CR}'],
                     color=colors_clipped[idx], linestyle='-', marker='*',
                     linewidth=2, markersize=8, label=f'Clipped CR={CR}')

    # Plot clipped+filtered curves
    for idx, CR in enumerate(CR_list):
        plt.semilogy(p["EbNo_range"], BER_results[f'clipped_filtered_CR={CR}'],
                     color=colors_filtered[idx], linestyle=':', marker='o',
                     linewidth=2, markersize=6, label=f'Clipped+Filtered CR={CR}')

    # Plot no clipping (simulated)
    plt.semilogy(p["EbNo_range"], BER_results['no_clipping'],
                 'o-', color='gray', linewidth=2, markersize=8,
                 label='No clipping (simulated)')

    # Plot theoretical
    plt.semilogy(p["EbNo_range"], BER_results['theoretical'],
                 'k-', linewidth=2.5, label='Theoretical')

    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title(f'BER vs SNR - {p["mod_str"]} with Clipping (L={L})', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.xlim([p["EbNo_range"][0], p["EbNo_range"][-1]])
    plt.ylim([1e-3, 1e0])
    plt.tight_layout()
    plt.show()

    # ----------------- PAPR simulation with clipping/filtering -----------------
    # calc_papr = input("\nDo you want to calculate PAPR and plot CCDF? (Enter 'Y' to continue and any other key to terminate): ").strip().lower()
    calc_papr = 'y'
    # ... inside main() ...
    print(calc_papr)
    if calc_papr == 'y' or calc_papr == '':
        # L_values = [1, 2, 4]
        L_values = [8]

        # --- FIX 1: ADJUST CR ---
        # A CR of 1.2 linear is ~1.6dB (Too low).
        # Let's use 7 dB which is a standard practical value.
        # target_CR_dB = 4.0
        # CR = 10 ** (target_CR_dB / 20)
        # CR = 0.7
        cr_db_list = [20 * np.log10(c) for c in CR_list]
        # --- FIX 2: REDUCE SAMPLES FOR SPEED ---
        # samples_per_L = 1024
        samples_per_L = 100000
        # print(f"Calculating PAPR for {samples_per_L} blocks/L with CR={20 * np.log10(CR)}dB ({CR:.2f} linear)...")
        print(f"Calculating PAPR for {samples_per_L} blocks/L with CRs (dB) = {cr_db_list} and linear = {CR_list}")

        for L in L_values:
            print(f"  Simulating L={L} ...")
            papr_orig = []
            papr_clipped = {cr: [] for cr in CR_list}
            papr_clipped_filtered = {cr: [] for cr in CR_list}

            mid = subc // 2
            zeros = np.zeros((L - 1) * subc, dtype=complex)


            # To limit time, we generate random frequency-domain OFDM symbols repeatedly
            for _ in range(samples_per_L):
                papr_bits = np.random.randint(0, 2, int(subc * bits_per_symbol))
                papr_symbols = modem.modulate(papr_bits)  # length = subc (frequency bins)

                # Compute oversampled time and apply clipping+filtering pipeline
                # Use function clip_and_filter_ofdm which returns the original oversampled time,
                # the clipped time (no filtering) and the clipped+filtered time.
                # tx_time_os, clipped_time, clipped_filtered_time = clip_and_filter_ofdm(papr_symbols, subc, L=L, CR=CR)

                oversampled_freq = np.concatenate([papr_symbols[:mid], zeros, papr_symbols[mid:]])
                tx_time_os = np.fft.ifft(oversampled_freq)
                papr_orig.append(PAPR_from_time(tx_time_os))

                # calculate PAPR for each
                # papr_orig.append(PAPR_from_time(tx_time_os))
                # papr_clipped.append(PAPR_from_time(clipped_time))
                # papr_clipped_filtered.append(PAPR_from_time(clipped_filtered_time))

                for cr in CR_list:
                    clipped_time = soft_clip_time(tx_time_os, cr)
                    clipped_freq = np.fft.fft(clipped_time)
                    kept_freq = np.zeros_like(clipped_freq)
                    kept_freq[:mid] = clipped_freq[:mid]
                    kept_freq[-mid:] = clipped_freq[-mid:]
                    clipped_filtered_time = np.fft.ifft(kept_freq)

                    papr_clipped[cr].append(PAPR_from_time(clipped_time))
                    papr_clipped_filtered[cr].append(PAPR_from_time(clipped_filtered_time))

            max_diff = 0.0
            for cr in CR_list:
                a = np.array(papr_clipped_filtered[cr])
                b = np.array(papr_clipped[cr])
                if a.size and b.size:
                    max_diff = max(max_diff, float(np.max(np.abs(a - b))))
            print(f"Max abs diff between clipped_filtered and clipped (across CRs): {max_diff:.6e}")

            papr_dict = {'Original (no clipping)': np.array(papr_orig)}
            for cr in CR_list:
                papr_dict[f'Clipped (CR={cr})'] = np.array(papr_clipped[cr])
                papr_dict[f'Clipped + Filtered (CR={cr})'] = np.array(papr_clipped_filtered[cr])

            plot_CCDF_compare(papr_dict, subc)

        print("PAPR simulation done.")
    else:
        print("Terminating.")

if __name__ == "__main__":
    main()

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")