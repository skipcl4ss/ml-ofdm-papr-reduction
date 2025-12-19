import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.special import erfc
import matplotlib.cm as cm
import sys

# # clipping.py

N = 2560
M = 16
L = 4
# N_iter = 5
CR = 1.5
# num_symbols = 10000
num_symbols = 1000
filter_order = 64

def make_filter():
    return firwin(filter_order + 1, 1 / L)

def qam_mod(bits):
    m = int(np.log2(M))
    gray = bits.reshape(-1, m)
    k = gray[:, 0] * 8 + gray[:, 1] * 4 + gray[:, 2] * 2 + gray[:, 3] * 1
    r = 2 * ((k % 4) - 1.5)
    i = 2 * ((k // 4) - 1.5)
    x = (r + 1j * i) / np.sqrt(10)
    return x

def ofdm_mod(symbols):
    symbols_oversampled = np.concatenate([symbols, np.zeros((L - 1) * N)])
    return np.fft.ifft(symbols_oversampled)

def papr(x):
    return 10 * np.log10(np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2))

def clip_signal(x, CR):
    A = CR * np.sqrt(np.mean(np.abs(x) ** 2))
    eps = 1e-12
    return np.where(np.abs(x) > A, A * x / (np.abs(x) + eps), x)

def filter_signal(x, h):
    return lfilter(h, 1, x)

def ccdf(values):
    values_sorted = np.sort(values)
    ccdf_vals = 1 - np.arange(len(values_sorted)) / len(values_sorted)
    return values_sorted, ccdf_vals

def clipping():
    papr_original = []
    papr_iter = []

    h = make_filter()

    for _ in range(num_symbols):
        bits = np.random.randint(0, 2, N * int(np.log2(M)))
        qam = qam_mod(bits)
        s = ofdm_mod(qam)

        papr_original.append(papr(s))

        x = s.copy()

        x = clip_signal(x, CR)
        x = filter_signal(x, h)
        papr_iter.append(papr(x))

    orig_mean = np.mean(papr_original)
    orig_median = np.median(papr_original)
    orig_max = np.max(papr_original)

    print("Original Mean PAPR (dB): ", orig_mean)
    print("Original Median PAPR (dB): ", orig_median)
    print("Original Max PAPR (dB): ", orig_max)

    mean_papr = np.mean(papr_iter)
    median_papr = np.median(papr_iter)
    max_papr = np.max(papr_iter)

    print("Mean PAPR after one iteration (dB): ", mean_papr)
    print("Median PAPR after one iteration (dB): ", median_papr)
    print("Max PAPR after one iteration (dB): ", max_papr)

    iteration = np.arange(2)

    plt.figure(figsize=(10, 6))
    plt.plot(iteration, [orig_mean, mean_papr], 'o-', label="Mean PAPR")
    plt.plot(iteration, [orig_median, median_papr], 's-', label="Median PAPR")
    plt.plot(iteration, [orig_max, max_papr], 'd-', label="Max PAPR")

    plt.axhline(orig_mean, color='gray', linestyle='--', label="Original Mean")
    plt.axhline(orig_median, color='gray', linestyle='-.', label="Original Median")
    plt.axhline(orig_max, color='gray', linestyle=':', label="Original Max")

    plt.xlabel("Iteration Number")
    plt.ylabel("PAPR (dB)")
    plt.title("PAPR Statistics vs Clipping–Filtering Iterations")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))

    papr_o, ccdf_o = ccdf(papr_original)
    plt.semilogy(papr_o, ccdf_o, label="Original")

    pi, ci = ccdf(papr_iter)
    plt.semilogy(pi, ci, label=f"Iteration 1")

    plt.grid(True, which='both')
    plt.xlabel("PAPR (dB)")
    plt.ylabel("CCDF")
    plt.title("PAPR CCDF – One Iteration Clipping & Filtering")
    plt.legend()
    plt.show()

# # ahmed.py

# todo implement param in clipping too
def param():
    print("Enter the OFDM Parameters")
    while True:
        try:
            N = int(input("Enter the number of subcarriers (Default: 64): ") or "64")
            cp = N // 4
            break
        except ValueError:
            print("Kindly enter a valid number.")

    while True:
        mod_str = input("Enter the modulation type [BPSK, QPSK, 8PSK, 16QAM, 64QAM] (Default: QPSK): ") or "QPSK"
        mod_dic = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64}
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
        "N": N,
        "cp": cp,
        "mod_str": mod_str.upper(),
        "M": M,
        "bits_per_symbol": bits_per_symbol,
        "EbNo_range": np.arange(EbNo_start, EbNo_end+EbNo_step, EbNo_step),
        "num_symb": num_symb
    }
    return parameters

# ? what does it do exactly
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

# ? what does it do exactly
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

def AWGN(tx_signal, SNR_dB):
    signal_power = np.mean(np.abs(tx_signal)**2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    return tx_signal + noise

# ! check if this is the best theoretical implementation
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

def soft_clip_time(tx_time, CR):
    rms = np.sqrt(np.mean(np.abs(tx_time)**2))
    A = CR * rms
    mag = np.abs(tx_time)
    phase = np.angle(tx_time)
    clipped = np.where(mag <= A, tx_time, A * np.exp(1j * phase)) # ? why
    return clipped

def clip_and_filter_ofdm(freq_symbols, N, L=4, CR=2.23):
    mid = N // 2
    zeros = np.zeros((L - 1) * N, dtype=complex)
    oversampled_freq = np.concatenate([freq_symbols[:mid], zeros, freq_symbols[mid:]])
    tx_time_oversampled = np.fft.ifft(oversampled_freq)
    clipped_time = soft_clip_time(tx_time_oversampled, CR)
    clipped_freq = np.fft.fft(clipped_time)
    kept_freq = np.zeros_like(clipped_freq)
    kept_freq[:mid] = clipped_freq[:mid]
    kept_freq[-mid:] = clipped_freq[-mid:]
    clipped_filtered_time = np.fft.ifft(kept_freq)
    return tx_time_oversampled, clipped_time, clipped_filtered_time

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

# todo align the parameters with those of clipping.py
def ahmed():
    p = param()
    N = p["N"]
    cp = p["cp"]
    M = p["M"]
    bits_per_symbol = p["bits_per_symbol"]
    num_symb = p["num_symb"]

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
            tx_bits = np.random.randint(0, 2, int(N * bits_per_symbol))
            tx_symbols = modem.modulate(tx_bits)
            tx_ofdm = np.fft.ifft(tx_symbols, N)
            tx_signal = np.concatenate([tx_ofdm[-cp:], tx_ofdm])
            rx_signal = AWGN(tx_signal, SNR_dB)

            rx_ofdm = rx_signal[cp:]
            rx_symbols = np.fft.fft(rx_ofdm, N)
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

    calc_papr = input("\nDo you want to calculate PAPR and plot CCDF? (Enter 'Y' to continue and any other key to terminate): ").strip().lower()
    if calc_papr == 'y':
        L_values = [1, 2, 4]

        # target_CR_dB = 7.0
        target_CR_dB = 4.0
        # target_CR_dB = 40 * np.log10(2)
        CR = 10 ** (target_CR_dB / 20)

        samples_per_L = 100000

        print(f"Calculating PAPR for {samples_per_L} blocks/L with CR={target_CR_dB}dB ({CR:.2f} linear)...")

        for L in L_values:
            print(f"  Simulating L={L} ... (this may take some time)")
            papr_orig = []
            papr_clipped = []
            papr_clipped_filtered = []

            for _ in range(samples_per_L):
                papr_bits = np.random.randint(0, 2, int(N * bits_per_symbol))
                papr_symbols = modem.modulate(papr_bits)

                tx_time_os, clipped_time, clipped_filtered_time = clip_and_filter_ofdm(papr_symbols, N, L=L, CR=CR)

                papr_orig.append(PAPR_from_time(tx_time_os))
                papr_clipped.append(PAPR_from_time(clipped_time))
                papr_clipped_filtered.append(PAPR_from_time(clipped_filtered_time))

            papr_dict = {
                'Original (no clipping)': np.array(papr_orig),
                f'Clipped (CR={CR})': np.array(papr_clipped),
                f'Clipped + Filtered (CR={CR})': np.array(papr_clipped_filtered),
            }
            plot_CCDF_compare(papr_dict, N, title_suffix=f' (L={L})')

        print("PAPR simulation done.")
    else:
        print("Terminating.")

# # shorter.py

# N = 2560
cp = N // 4

def mapping(bits, modulation_type):
    if  modulation_type == '16QAM':
        mapping_table = {
            (0, 0, 0, 0): -3 - 3j,
            (0, 0, 0, 1): -3 - 1j,
            (0, 0, 1, 0): -3 + 3j,
            (0, 0, 1, 1): -3 + 1j,
            (0, 1, 0, 0): -1 - 3j,
            (0, 1, 0, 1): -1 - 1j,
            (0, 1, 1, 0): -1 + 3j,
            (0, 1, 1, 1): -1 + 1j,
            (1, 0, 0, 0): 3 - 3j,
            (1, 0, 0, 1): 3 - 1j,
            (1, 0, 1, 0): 3 + 3j,
            (1, 0, 1, 1): 3 + 1j,
            (1, 1, 0, 0): 1 - 3j,
            (1, 1, 0, 1): 1 - 1j,
            (1, 1, 1, 0): 1 + 3j,
            (1, 1, 1, 1): 1 + 1j
        }
        modulated_signal = np.array([mapping_table[tuple(b)] for b in bits])
        norm_factor = np.sqrt(np.mean(np.abs(modulated_signal) ** 2))
        modulated_signal = modulated_signal / norm_factor
        normalized_mapping_table = {v / norm_factor: k for k, v in mapping_table.items()}
        demapping_table = normalized_mapping_table
        pass
    else:
        raise ValueError("Unsupported modulation type")

    return modulated_signal, demapping_table

def addCP(OFDM_time):
    CP = OFDM_time[-cp:]
    return np.hstack([CP, OFDM_time])

def channel(signal, SNRdb):
    signal_power = np.mean(abs(signal)**2)
    sigma2 = signal_power * 10 ** (-SNRdb / 10)

    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    return signal + noise

def removeCP(signal):
    return signal[cp:(cp+N)]

def Demapping(QAM, demapping_table):
    constellation = np.array([x for x in demapping_table.keys()])

    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

    const_index = dists.argmin(axis=1)

    hardDecision = constellation[const_index]

    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def theoretical_ber(modulation_type, Eb_No):
    if modulation_type == '16QAM':
        ber_theoretical = (3/8) * erfc(np.sqrt(0.4 * Eb_No))

    else:
        raise ValueError("Unsupported modulation type")

    return ber_theoretical

def shorter():
    modulation_type = '16QAM'

    if modulation_type == '16QAM':
        mu = 4
        SNR_max = 20
        SNR_min = -10
        SNR_step = 1

    else:
        print("Invalid input detected, stopping execution")
        sys.exit()

    # N = 2560
    # cp = N // 4
    dataCarriers = np.arange(N)
    payloadBits_per_signal = len(dataCarriers) * mu

    K_PAPR = 1000
    PAPR_values = []

    print(f"Simulating PAPR with {K_PAPR} symbols...")

    for _ in range(K_PAPR):
        bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
        bits_SP = bits.reshape((len(dataCarriers), mu))

        mapped_bits, _ = mapping(bits_SP, modulation_type)

        OFDM_time = np.fft.ifft(mapped_bits)

        peak_power = np.max(np.abs(OFDM_time) ** 2)
        avg_power = np.mean(np.abs(OFDM_time) ** 2)

        PAPR = peak_power / avg_power
        PAPR_values.append(PAPR)

    PAPR_dB = 10 * np.log10(PAPR_values)
    PAPR_dB = np.array(PAPR_dB)

    PAPR_dB_range = np.linspace(np.min(PAPR_dB), np.max(PAPR_dB), 100)
    papr_linear = 10 ** (PAPR_dB_range / 10)
    CCDF_theoretical = 1 - (1 - np.exp(-papr_linear)) ** N

    CCDF = [np.mean(PAPR_dB > t) for t in PAPR_dB_range]

    K_BER = 1  # Number of OFDM frames for BER (Increase for smoother BER curves)
    SNR_dB = np.arange(SNR_min, SNR_max, SNR_step)

    plot_snr_values = [-10, 0, 10]
    saved_symbols = {}
    BER = []

    print(f"Simulating BER with {K_BER} symbols per SNR step...")

    for snr in SNR_dB:
        ber_sum = 0
        for n in range(K_BER):
            bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
            bits_SP = bits.reshape((len(dataCarriers), mu))

            mapped_bits, demapping_table = mapping(bits_SP, modulation_type)
            OFDM_time = np.fft.ifft(mapped_bits)

            OFDM_CP = addCP(OFDM_time)

            OFDM_TX = OFDM_CP
            OFDM_RX = channel(OFDM_TX, snr)

            OFDM_RX_noCP = removeCP(OFDM_RX)
            OFDM_demod = np.fft.fft(OFDM_RX_noCP)
            symbols_est = OFDM_demod[dataCarriers]

            if snr in plot_snr_values and n == 0:
                saved_symbols[snr] = {
                    'transmitted': mapped_bits,
                    'received': symbols_est,
                }

            bits_demapped, hardDecision = Demapping(symbols_est, demapping_table)
            bits_est = bits_demapped.reshape((-1,))

            bit_errors = np.sum(bits != bits_est)
            ber = bit_errors / len(bits)
            ber_sum += ber

        BER.append(ber_sum / K_BER)


    plt.figure()
    plt.semilogy(PAPR_dB_range, CCDF, 'b-', lw=2, label='Simulated CCDF')
    plt.semilogy(PAPR_dB_range, CCDF_theoretical, 'r--', label='Theoretical CCDF')
    plt.grid(True, which='both', alpha=0.5)
    plt.xlabel('PAPR (dB)')
    plt.ylabel('Pr(PAPR > PAPR0)')
    plt.title(f'CCDF of PAPR for {modulation_type} (N={N})')
    plt.legend()

    SNR_linear = 10 ** (SNR_dB / 10)
    Eb_No = SNR_linear / mu
    Eb_No_dB = 10 * np.log10(Eb_No)
    BER_theoretical = theoretical_ber(modulation_type, Eb_No)

    plt.figure()
    plt.semilogy(Eb_No_dB, BER, 'bo-', label='Simulated BER')
    plt.semilogy(Eb_No_dB, BER_theoretical, 'r--', label='Theoretical BER')
    plt.title(f"Bit Error Rate (BER) vs SNR ({modulation_type})")
    plt.xlabel("Eb/No (dB)")
    plt.ylabel("BER")
    plt.grid(True, which='both', alpha=0.5)
    plt.ylim(bottom=1e-5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clipping()
    # ahmed()
    # shorter()