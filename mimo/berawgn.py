import numpy as np
from scipy.special import erfc

# implementation of the berawgn MATLAB line in CCDF_of_clipped_filtered_OFDM_signal.m
def berawgn_qam(EbN0_dB, M):
    """Analytical BER for QAM in AWGN channel"""
    EbN0 = 10 ** (EbN0_dB / 10)
    k = np.sqrt(M)
    ber = 2 * (1 - 1 / k) / np.log2(M) * 0.5 * erfc(np.sqrt(3 * np.log2(M) / (2 * (M - 1)) * EbN0))
    return ber