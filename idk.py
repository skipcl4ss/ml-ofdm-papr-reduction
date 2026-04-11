import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
import time

start = time.time()

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

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

def channel(signal, SNRdb):
    signal_power = np.mean(abs(signal)**2)
    sigma2 = signal_power * 10 ** (-SNRdb / 10)
    #print("TX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    return signal + noise

def removeCP(signal):
    return signal[CP:(CP+N)]

def Demapping(QAM, demapping_table):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)

    # get back the real constellation point
    hardDecision = constellation[const_index]

    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def theoretical_ber(modulation_type, Eb_No):
    if modulation_type == '16QAM':
        ber_theoretical = (3/8) * erfc(np.sqrt(0.4 * Eb_No))
        pass

    else:
        raise ValueError("Unsupported modulation type")

    return ber_theoretical


if __name__ == "__main__":

    modulation_type = '16QAM'  # choose modulation type (QPSK/16-QAM/64-QAM)

    # Configuration
    if modulation_type == '16QAM':
        mu = 4
        SNR_max = 20
        SNR_min = -10
        SNR_step = 1

    else:
        print("Invalid input detected, stopping execution")
        sys.exit()

    # Simulation Parameters
    N = 256  # number of OFDM subcarriers
    CP = N // 4  # length of the cyclic prefix
    dataCarriers = np.arange(N)
    payloadBits_per_signal = len(dataCarriers) * mu

    # --- PHASE 1: PAPR Simulation (No Channel/Noise) ---
    # We use a high number of symbols here to get a smooth CCDF curve
    K_PAPR =100000
    PAPR_values = []

    print(f"Simulating PAPR with {K_PAPR} symbols...")

    for _ in range(K_PAPR):
        # Generate random bits
        bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_signal)
        bits_SP = bits.reshape((len(dataCarriers), mu))

        # Map bits (we only need the mapping here, not the demapping table)
        mapped_bits, _ = mapping(bits_SP, modulation_type)

        # IFFT to time domain
        OFDM_time = np.fft.ifft(mapped_bits)

        # Calculate PAPR for this symbol
        # Power calculations
        peak_power = np.max(np.abs(OFDM_time) ** 2)
        avg_power = np.mean(np.abs(OFDM_time) ** 2)

        PAPR = peak_power / avg_power
        PAPR_values.append(PAPR)

    # Process PAPR Statistics
    PAPR_dB = 10 * np.log10(PAPR_values)
    PAPR_dB = np.array(PAPR_dB)

    PAPR_dB_range = np.linspace(np.min(PAPR_dB), np.max(PAPR_dB), 100)
    # Calculate theoretical CCDF
    papr_linear = 10 ** (PAPR_dB_range / 10)
    # Correct theoretical approximation for large N
    CCDF_theoretical = 1 - (1 - np.exp(-papr_linear)) ** N

    # Calculate empirical CCDF
    CCDF = [np.mean(PAPR_dB > t) for t in PAPR_dB_range]

    # --- PHASE 2: BER Simulation (With Channel/Noise) ---
    K_BER = 100  # Number of OFDM frames for BER (Increase for smoother BER curves)
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

            # Adding cyclic prefix
            OFDM_CP = addCP(OFDM_time)

            # Channel
            OFDM_TX = OFDM_CP
            OFDM_RX = channel(OFDM_TX, snr)

            # Receiver
            OFDM_RX_noCP = removeCP(OFDM_RX)
            OFDM_demod = np.fft.fft(OFDM_RX_noCP)
            symbols_est = OFDM_demod[dataCarriers]

            # Save constellation for plotting
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

    # --- PLOTTING ---

    # 1. PAPR Plot
    plt.figure()
    plt.semilogy(PAPR_dB_range, CCDF, 'b-', lw=2, label='Simulated CCDF')
    plt.semilogy(PAPR_dB_range, CCDF_theoretical, 'r--', label='Theoretical CCDF')
    plt.grid(True, which='both', alpha=0.5)
    plt.xlabel('PAPR (dB)')
    plt.ylabel('Pr(PAPR > PAPR0)')
    plt.title(f'CCDF of PAPR for {modulation_type} (N={N})')
    plt.legend()

    # 2. BER Plot
    SNR_linear = 10 ** (SNR_dB / 10)
    Eb_No = SNR_linear / mu
    Eb_No_dB = 10 * np.log10(Eb_No)
    BER_theoretical = theoretical_ber(modulation_type, Eb_No)

    plt.figure()
    plt.semilogy(Eb_No_dB, BER, 'bo-', label='Simulated BER')
    plt.semilogy(Eb_No_dB, BER_theoretical, 'r--', label='Theoretical BER')
    plt.title(f"Bit Error Rate (BER) vs SNR ({modulation_type}) (idk.py)")
    plt.xlabel("Eb/No (dB)")
    plt.ylabel("BER")
    plt.grid(True, which='both', alpha=0.5)
    plt.ylim(bottom=1e-5)  # Limit y-axis to avoid -inf issues
    plt.legend()
    plt.tight_layout()
    plt.show()

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")
