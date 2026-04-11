import numpy as np
import matplotlib.pyplot as plt
from modem import PSKModem, QAMModem, AWGN, TBER
import matplotlib.cm as cm
import time

start = time.time()

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

def PAPR(tx_freq_symbols, N, L=4):
    zeros = np.zeros((L - 1) * N, dtype=complex)
    mid = N // 2
    oversampled_input = np.concatenate([tx_freq_symbols[:mid], zeros, tx_freq_symbols[mid:]])
    tx_time = np.fft.ifft(oversampled_input)
    power = np.abs(tx_time)**2
    peak = np.max(power)
    avg = np.mean(power)

    if avg == 0: return 0
    return 10 * np.log10(peak / avg)

def theoretical_CCDF(N, papr_db_range):
    gamma = 10 ** (papr_db_range / 10)
    ccdf = 1 - (1 - np.exp(-gamma))**N
    return ccdf

def plot_CCDF(results_dict, N):
    plt.figure(figsize=(10, 7))
    colors = cm.viridis(np.linspace(0, 0.9, len(results_dict)))

    for idx, (L, papr_data) in enumerate(results_dict.items()):
        sorted_papr = np.sort(papr_data)
        y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr)
        plt.semilogy(sorted_papr, y_axis, color=colors[idx], linewidth=2, label=f'L = {L}')

    if len(results_dict) > 0:
        all_data = np.concatenate(list(results_dict.values()))
        x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
        y_theory = theoretical_CCDF(N, x_theory)
        plt.semilogy(x_theory, y_theory, 'k--', linewidth=2, label='Theoretical (L=1)')

    plt.xlabel('PAPR Threshold ($PAPR_0$) [dB]')
    plt.ylabel('Probability ($Pr(PAPR > PAPR_0)$)')
    plt.title(f'CCDF of PAPR (No. of subcarriers = {N})')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = param()
    subc = p["subc"]
    cp = p["cp"]
    M = p["M"]
    bits_per_symbol = p["bits_per_symbol"]
    num_symb = p["num_symb"]

    BER = []
    BER_theory = []
    print(f"\nStarting Simulation for {p['mod_str']}...")
    if p["M"] in [2, 4, 8]:
        modem = PSKModem(M)
    else:
        modem = QAMModem(M)

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

    plt.figure(figsize=(10, 6))
    plt.semilogy(p["EbNo_range"], BER, 'bo-', label=f'Simulated ({p["mod_str"]})', linewidth=2, markersize=8)
    plt.semilogy(p["EbNo_range"], BER_theory, 'r--', label=f'Theoretical ({p["mod_str"]})', linewidth=2)
    plt.xlabel('Eb/No (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(f'OFDM BER Performance - {p["mod_str"]} Modulation (ofdmyoussef.py)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

calc_papr = input("\nDo you want to calculate PAPR and plot CCDF? (Enter 'Y' to continue and any other key to terminate): ").strip().lower()

if calc_papr == 'y':
    L_values = [1, 2, 4]
    print(f"Calculating PAPR for 500,000 symbols for L = {L_values}...")

    papr_results_dict = {}

    for L in L_values:
        print(f"  Simulating L={L}...")
        current_papr_list = []
        for _ in range(100000):
            papr_bits = np.random.randint(0, 2, int(subc * bits_per_symbol))
            papr_symbols = modem.modulate(papr_bits)
            val = PAPR(papr_symbols, subc, L)
            current_papr_list.append(val)
        papr_results_dict[L] = current_papr_list

    plot_CCDF(papr_results_dict, subc)
    print("Done.")
else:
    print("Terminating.")

end = time.time()
# ~56s (with input)
print(f"\nTotal execution time: {end - start:.2f} seconds")
