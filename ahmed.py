# ofdm_with_clipping_filtering.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.cm as cm

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
def AWGN(tx_signal, SNR_dB):
    signal_power = np.mean(np.abs(tx_signal)**2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    return tx_signal + noise

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
        sorted_papr = np.sort(data)
        y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr)
        plt.semilogy(sorted_papr, y_axis, color=colors[idx], linewidth=2, label=label)

    all_data = np.concatenate(list(papr_dict.values()))
    x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    y_theory = theoretical_CCDF(N, x_theory)
    plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plt.xlabel('PAPR Threshold [dB]')
    plt.ylabel('Pr(PAPR > PAPR0)')
    plt.title(f'PAPR CCDF Comparison {title_suffix} (N={N})')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.tight_layout()
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
    clipped = np.where(mag <= A, tx_time, A * np.exp(1j * phase))
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
    # IFFT to get clipped+filtered time signal
    clipped_filtered_time = np.fft.ifft(kept_freq)
    return tx_time_oversampled, clipped_time, clipped_filtered_time

# ----------------- Parameter input (kept interactive like original) -----------------
def param():
    print("Enter the OFDM Parameters")

    while True:
        try:
            subc = int(input("Enter the number of subcarriers (Default: 64): ") or "64")
            cp = subc // 4
            break
        except ValueError:
            print("Kindly enter a valid number.")

    while True:
        mod_str = input("Enter the modulation type [BPSK, QPSK, 8PSK, 16QAM, 64QAM] (Default: QPSK): ") or "QPSK"
        mod_dic = {"BPSK":2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64}
        if mod_str.upper() in mod_dic:
            M = mod_dic[mod_str.upper()]
            bits_per_symbol = int(np.log2(M))
            break
        else:
            print("Kindly enter a valid modulation type.")

    while True:
        try:
            EbNo_start = int(input("Enter the start Eb/No (Default: 0): ") or "0")
            EbNo_end = int(input("Enter the stop Eb/No (Default: 10): " )or "10")
            EbNo_step = float(input("Enter the Eb/No step: (Default: 1): ") or "1")
            break
        except ValueError:
            print("Kindly enter a valid number.")

    while True:
        try:
            num_symb = int(input("Enter number of OFDM symbols (Default: 1000): ") or "1000")
            break
        except ValueError:
            print("Kindly enter a valid number")

    parameters = {
        "subc": subc,
        "cp": cp,
        "M": M,
        "EbNo_range": np.arange(EbNo_start, EbNo_end+EbNo_step, EbNo_step),
        "bits_per_symbol": bits_per_symbol,
        "num_symb": num_symb,
        "mod_str": mod_str.upper()
    }
    return parameters

# ----------------- Main simulation -----------------
def main():
    p = param()
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

    print(f"\nStarting Simulation for {p['mod_str']}...")
    BER = []
    BER_theory = []

    low_snr_idx = 0
    high_snr_idx = len(p["EbNo_range"]) - 1

    for i, EbNo_dB in enumerate(p["EbNo_range"]):
        SNR_dB = EbNo_dB + 10 * np.log10(bits_per_symbol)

        bit_error = 0
        total_bits = 0
        plotted_this_round = False

        for _ in range(num_symb):
            tx_bits = np.random.randint(0, 2, int(subc * bits_per_symbol))
            tx_symbols = modem.modulate(tx_bits)
            tx_ofdm = np.fft.ifft(tx_symbols, subc)
            tx_signal = np.concatenate([tx_ofdm[-cp:], tx_ofdm])
            rx_signal = AWGN(tx_signal, SNR_dB)

            rx_ofdm = rx_signal[cp:]
            rx_symbols = np.fft.fft(rx_ofdm, subc)
            rx_bits = modem.demodulate(rx_symbols, 'hard')

            bit_error += np.sum(tx_bits != rx_bits)
            total_bits += len(tx_bits)

            if not plotted_this_round:
                limit = max(np.max(np.abs(tx_symbols)), 1.5) * 1.1

                if i == low_snr_idx:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(tx_symbols.real, tx_symbols.imag, c='blue', marker='o', s=20, label='TX Symbols')
                    plt.title(f'Transmitted Constellation - {p["mod_str"]}')
                    plt.xlabel('In-Phase (I)')
                    plt.ylabel('Quadrature (Q)')
                    plt.grid(True)
                    plt.axhline(0, color='black', linewidth=1)
                    plt.axvline(0, color='black', linewidth=1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim(-limit, limit)
                    plt.ylim(-limit, limit)
                    plt.show()

                if i == low_snr_idx:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(rx_symbols.real, rx_symbols.imag, c='red', marker='o', s=20, label='RX Symbols')
                    plt.title(f'Received Constellation @ Low Eb/No ({EbNo_dB} dB)')
                    plt.xlabel('In-Phase (I)')
                    plt.ylabel('Quadrature (Q)')
                    plt.legend()
                    plt.grid(True)
                    plt.axhline(0, color='black', linewidth=1)
                    plt.axvline(0, color='black', linewidth=1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim(-limit, limit)
                    plt.ylim(-limit, limit)
                    plt.show()

                elif i == high_snr_idx:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(rx_symbols.real, rx_symbols.imag, c='red', marker='o', s=20, label='RX Symbols')
                    plt.title(f'Received Constellation @ High Eb/No ({EbNo_dB} dB)')
                    plt.xlabel('In-Phase (I)')
                    plt.ylabel('Quadrature (Q)')
                    plt.legend()
                    plt.grid(True)
                    plt.axhline(0, color='black', linewidth=1)
                    plt.axvline(0, color='black', linewidth=1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim(-limit, limit)
                    plt.ylim(-limit, limit)
                    plt.show()

                plotted_this_round = True

        ber = bit_error / total_bits
        ber_theory = TBER(EbNo_dB, M)

        BER.append(ber)
        BER_theory.append(ber_theory)
        print(f"  Eb/No: {EbNo_dB:.2f} dB | BER (Sim): {ber:.6f} | BER (Theory): {ber_theory:.6f}")

    # plot BER curves
    plt.figure(figsize=(10, 6))
    plt.semilogy(p["EbNo_range"], BER, 'bo-', label=f'Simulated ({p["mod_str"]})', linewidth=2, markersize=8)
    plt.semilogy(p["EbNo_range"], BER_theory, 'r--', label=f'Theoretical ({p["mod_str"]})', linewidth=2)
    plt.xlabel('Eb/No (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(f'OFDM BER Performance - {p["mod_str"]} Modulation', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ----------------- PAPR simulation with clipping/filtering -----------------
    calc_papr = input("\nDo you want to calculate PAPR and plot CCDF? (Enter 'Y' to continue and any other key to terminate): ").strip().lower()
    # ... inside main() ...
    if calc_papr == 'y':
        L_values = [1, 2, 4]

        # --- FIX 1: ADJUST CR ---
        # A CR of 1.2 linear is ~1.6dB (Too low).
        # Let's use 7 dB which is a standard practical value.
        target_CR_dB = 7.0
        CR = 10 ** (target_CR_dB / 20)

        # --- FIX 2: REDUCE SAMPLES FOR SPEED ---
        samples_per_L = 2000  # Reduced from 100,000 for standard testing

        print(f"Calculating PAPR for {samples_per_L} blocks/L with CR={target_CR_dB}dB ({CR:.2f} linear)...")

        for L in L_values:
            # ... rest of your loop ...
            print(f"  Simulating L={L} ... (this may take some time)")
            papr_orig = []
            papr_clipped = []
            papr_clipped_filtered = []

            # To limit time, we generate random frequency-domain OFDM symbols repeatedly
            for _ in range(samples_per_L):
                papr_bits = np.random.randint(0, 2, int(subc * bits_per_symbol))
                papr_symbols = modem.modulate(papr_bits)  # length = subc (frequency bins)

                # Compute oversampled time and apply clipping+filtering pipeline
                # Use function clip_and_filter_ofdm which returns the original oversampled time,
                # the clipped time (no filtering) and the clipped+filtered time.
                tx_time_os, clipped_time, clipped_filtered_time = clip_and_filter_ofdm(papr_symbols, subc, L=L, CR=CR)

                # calculate PAPR for each
                papr_orig.append(PAPR_from_time(tx_time_os))
                papr_clipped.append(PAPR_from_time(clipped_time))
                papr_clipped_filtered.append(PAPR_from_time(clipped_filtered_time))

            # store results and plot CCDF for this L
            papr_dict = {
                'Original (no clipping)': np.array(papr_orig),
                f'Clipped (CR={CR})': np.array(papr_clipped),
                f'Clipped + Filtered (CR={CR})': np.array(papr_clipped_filtered),
            }
            plot_CCDF_compare(papr_dict, subc, title_suffix=f' (L={L})')

        print("PAPR simulation done.")
    else:
        print("Terminating.")

if __name__ == "__main__":
    main()