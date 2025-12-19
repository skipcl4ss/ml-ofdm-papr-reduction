import sys

import numpy as np
import komm
from matplotlib import pyplot as plt
from scipy.special import erfc

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def removeCP(signal):
    return signal[CP:(CP+K)]

def theoretical_ber(modulation_type, Eb_No):
    if modulation_type == 'QPSK':
        ber_theoretical = 0.5 * erfc(np.sqrt(2*Eb_No)/np.sqrt(2))
        pass

    elif modulation_type == '16QAM':
        ber_theoretical = (3/8) * erfc(np.sqrt(0.4 * Eb_No))
        pass

    elif modulation_type == '64QAM':
        ber_theoretical = (7/24) * erfc(np.sqrt((1/7) * Eb_No))
        pass

    else:
        raise ValueError("Unsupported modulation type")

    return ber_theoretical


if __name__ == "__main__":

    modulation_type = '16QAM' # choose modulation type (QPSK/16-QAM/64-QAM)

    if modulation_type == 'QPSK':
        mu = 2
        constellation = komm.PSKConstellation(4, phase_offset= 1 / 8)
    elif modulation_type == '16QAM':
        mu = 4
        constellation = komm.QAMConstellation(16)
    elif modulation_type == '64QAM':
        mu = 6
        constellation = komm.QAMConstellation(64)
    else:
        print("Invalid input detected, stopping execution")
        sys.exit()


    N = 1000 # number of OFDM signals
    SNR_max = 20 # SNR range in dB (-SNR_max to SNR_max)
    SNR_step = 2
    K = 512  # number of OFDM subcarriers
    CP = K // 4  # length of the cyclic prefix: 25% of the block
    dataCarriers = np.arange(K)  # index of all subcarriers ([0, 1, ... K-1])
    payloadBits_per_OFDM = len(dataCarriers) * mu  # number of payload bits per OFDM symbol

    #print("dataCarriers:  %s" % dataCarriers)
    #plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
    #plt.legend()
    #plt.show()


    # draw_constellation(modulation_type)          # use to map QPSK/16-QAM/64-QAM constellations

    SNR_dB = np.arange(-SNR_max, SNR_max, SNR_step)  # From -SNR_max to SNR_max dB

    # initializing vars for BER calculation

    BER = []
    PAPR_values = []
    ber_sum = 0

    for snr in SNR_dB:
        for n in range(N):
            bits = np.random.binomial(n=1, p=0.5, size=payloadBits_per_OFDM)
            # bits = [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
            # bits = np.array(bits)
            #print("Bits count: ", len(bits))
            #print("First 20 bits: ", bits[:20])
            #print("Mean of bits (should be around 0.5): ", np.mean(bits))

            bits_SP = bits.reshape(len(dataCarriers), mu)
            bitsIndex = komm.bits_to_int(bits_SP) # index each (mu) bits
            symbols = constellation.indices_to_symbols(bitsIndex)
            symbols = symbols/np.sqrt(np.mean(np.abs(symbols) ** 2)) # normalizing power


            #print("First 5 symbols and bits:")
            #print(bits_SP[:5, :])
            #print(mapped_bits[:5])

            OFDM_data = symbols
            #print("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

            OFDM_time = np.fft.ifft(OFDM_data)
            #print("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

            peak_power = np.max(np.abs(OFDM_time) ** 2)
            avg_power = np.mean(np.abs(OFDM_time) ** 2)
            PAPR = peak_power / avg_power
            PAPR_db = 10 * np.log10(PAPR)
            PAPR_values.append(PAPR_db)

            OFDM_CP = addCP(OFDM_time)
            #print("Number of OFDM samples in time domain with CP: ", len(OFDM_CP))
            OFDM_TX = OFDM_CP

            snr_linear = 10 ** (snr/10)
            awgn = komm.AWGNChannel(signal_power = "measured", snr = snr_linear)
            OFDM_RX = awgn.transmit(OFDM_TX)


            # plotting TX and RX signal power
            # plt.figure(figsize=(8, 2))
            # plt.plot(abs(OFDM_TX), label='TX signal')
            # plt.plot(abs(OFDM_RX), label='RX signal')
            # plt.legend(fontsize=10)
            # plt.xlabel('Time')
            # plt.ylabel('$|x(t)|$')
            # plt.grid(True)
            # plt.show()


            OFDM_RX_noCP = removeCP(OFDM_RX)
            OFDM_demod = np.fft.fft(OFDM_RX_noCP)



            index_est = constellation.closest_indices(OFDM_demod)
            #plt.plot(symbols_est.real, symbols_est.imag, 'bo')
            #plt.show()

            bits_rx = komm.int_to_bits(index_est, mu) # each integer is converted to (mu) bits
            bits_rx = bits_rx.reshape(1, -1)

            #for qam, hard in zip(symbols_est, hardDecision):
                #plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
                #plt.plot(hardDecision.real, hardDecision.imag, 'ro')
            #plt.show()

            #print("First 10 mapped symbols:", mapped_bits[:10])
            #print("First 10 received symbols:", symbols_est[:10])
            #print("First 10 hard decisions:", hardDecision[:10])
            #for i in range(10):
            #    print("Mapped bits:", bits_SP[i], "| Demapped bits:", bits_demapped[i])


            bit_errors = np.sum(bits != bits_rx)
            ber = bit_errors / len(bits)
            ber_sum = ber_sum + ber

        ber_sum = ber_sum / N
        BER.append(ber_sum)
        ber_sum = 0


    SNR_linear = 10 ** (SNR_dB / 10)
    Eb_No = SNR_linear/mu # mu is no. of bits per symbol
    Eb_No_dB = 10 * np.log10(Eb_No)
    BER_theoretical = theoretical_ber(modulation_type, Eb_No)

    PAPR_values = np.array(PAPR_values)

    PAPR_dB_range = np.linspace(np.min(PAPR_values), np.max(PAPR_values), 100)
    papr_linear = 10 ** (PAPR_dB_range / 20)
    CCDF = [np.mean(PAPR_values > t) for t in PAPR_dB_range]
    CCDF_theoretical = 1 - (1 - np.exp(-papr_linear**2))**K


    plt.figure()
    plt.semilogy(PAPR_dB_range, CCDF, 'b-', lw = 2, label='CCDF')
    plt.semilogy(PAPR_dB_range, CCDF_theoretical, 'r--', label='CCDF_theoretical')
    plt.xlim((2, 13))
    plt.ylim((10 ** -4, 10 ** 0))
    plt.grid(True, which='both')
    plt.xlabel('PAPR (dB)')
    plt.ylabel('CCDF')
    plt.title('CCDF of PAPR for OFDM (testnour.py)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(Eb_No_dB, BER, 'bo-', label='Simulated BER')
    plt.semilogy(Eb_No_dB, BER_theoretical, 'r--', label='Theoretical BER')
    plt.title("Bit Error Rate (BER) vs SNR")
    plt.xlabel("Eb/No")
    plt.ylabel("BER")
    plt.xlim(-20, 15)
    plt.ylim(10**-6, 10**0)
    plt.grid(True, which='both')
    plt.legend()
    plt.show()