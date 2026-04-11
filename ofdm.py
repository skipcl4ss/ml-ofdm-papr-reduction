"""
Complete OFDM Transceiver Simulator
Includes:
- QAM/BPSK modulation & demodulation
- OFDM modulation/demodulation with CP
- AWGN channel
- BER vs Eb/N0 simulation
- Theoretical BER curves (BPSK & M-QAM)
- PAPR computation per OFDM symbol
- PAPR CCDF curve
"""

import numpy as np
import matplotlib.pyplot as plt
from math import erfc
import time

start = time.time()

#############################
# Modulation / Demodulation #
#############################

def qam_mod(bits, M):
    k = int(np.log2(M))
    assert bits.size % k == 0
    bit_groups = bits.reshape((-1, k)).astype(np.int64, copy=False)

    m_side = int(np.sqrt(M))
    assert m_side*m_side == M
    levels = np.arange(m_side)*2 - (m_side-1)

    half = k // 2
    weights = (1 << np.arange(half - 1, -1, -1, dtype=np.int64))
    i_idx = bit_groups[:, :half] @ weights
    q_idx = bit_groups[:, half:] @ weights
    symbols = levels[i_idx] + 1j * levels[q_idx]

    symbols /= np.sqrt(np.mean(np.abs(symbols)**2))
    return symbols


def qam_demod(symbols, M):
    k = int(np.log2(M))
    m_side = int(np.sqrt(M))
    levels = np.arange(m_side)*2 - (m_side-1)

    # Renormalize
    grid = np.array([x+1j*y for x in levels for y in levels])
    norm = np.sqrt(np.mean(np.abs(grid)**2))
    symbols = symbols * norm

    i_idx = np.argmin(np.abs(np.real(symbols)[:, None] - levels[None, :]), axis=1)
    q_idx = np.argmin(np.abs(np.imag(symbols)[:, None] - levels[None, :]), axis=1)

    half = k // 2
    shifts = np.arange(half - 1, -1, -1, dtype=np.int64)
    i_bits = ((i_idx[:, None] >> shifts) & 1)
    q_bits = ((q_idx[:, None] >> shifts) & 1)
    return np.concatenate([i_bits, q_bits], axis=1).reshape(-1)

#########################
# OFDM Mod/Demod + PAPR #
#########################

def ofdm_mod(symbols, N, cp_len):
    num_ofdm = int(np.ceil(len(symbols)/N))
    padded = np.zeros(num_ofdm*N, dtype=complex)
    padded[:len(symbols)] = symbols

    blocks = padded.reshape(num_ofdm, N)
    x = np.fft.ifft(blocks, axis=1) * np.sqrt(N)
    cp = x[:, -cp_len:]
    out = np.concatenate([cp, x], axis=1)

    power = np.abs(x) ** 2
    papr_vals = 10 * np.log10(np.max(power, axis=1) / np.mean(power, axis=1))

    return out.reshape(-1), num_ofdm, papr_vals.tolist()


def ofdm_demod(rx, N, cp_len, num_ofdm):
    L = N + cp_len
    rx_blocks = rx[:num_ofdm * L].reshape(num_ofdm, L)
    x = rx_blocks[:, cp_len:]
    fd = np.fft.fft(x, axis=1) / np.sqrt(N)
    return fd.reshape(-1)

#############
# Channel   #
#############

def awgn(signal, snr_db):
    P = np.mean(np.abs(signal)**2)
    snr = 10**(snr_db/10)
    N = P/snr
    noise = np.sqrt(N/2)*(np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))
    return signal+noise

###########
# Helpers #
###########

def bits_to_random(n):
    return np.random.randint(0,2,n)


def ber(a,b):
    L = min(len(a),len(b))
    return np.mean(a[:L]!=b[:L])

###########
# Theory  #
###########

def Q(x): return 0.5*erfc(x/np.sqrt(2))

def ber_theory_mqam(ebno_db, M):
    if M==2:  # BPSK
        x = np.sqrt(2*10**(ebno_db/10))
        return Q(x)
    k = np.log2(M)
    term = np.sqrt((3*k*10**(ebno_db/10))/(M-1))
    return (4*(1-1/np.sqrt(M))/k)*Q(term)

###############################
# Main Simulation Entry Block #
###############################

if __name__ == '__main__':

    # User parameters
    N = 512
    cp_len = 16
    M = 16  # set M=2 for BPSK
    num_blocks = 100000

    k = int(np.log2(M))
    total_symbols = N*num_blocks
    total_bits = total_symbols*k

    # Generate bits + QAM
    tx_bits = bits_to_random(total_bits)
    tx_syms = qam_mod(tx_bits, M)

    # OFDM modulation + PAPR
    tx_time, num_ofdm, papr_vals = ofdm_mod(tx_syms, N, cp_len)

    # Normalize TX power
    tx_time = tx_time/np.sqrt(np.mean(np.abs(tx_time)**2))

    # Eb/N0 range
    ebno_db_range = np.arange(0,16,2)
    ber_vals = []

    rate = k * (N/(N+cp_len))

    print("Starting loop")
    for ebno_db in ebno_db_range:
        print(f"Iteration {ebno_db // 2 + 1}")
        snr_db = ebno_db + 10*np.log10(rate)
        rx_time = awgn(tx_time, snr_db)
        rx_syms = ofdm_demod(rx_time, N, cp_len, num_ofdm)
        rx_bits = qam_demod(rx_syms[:len(tx_syms)], M)
        ber_vals.append(ber(tx_bits, rx_bits))

    # ---- BER Plot (Sim + Theory) ----
    plt.figure()
    plt.semilogy(ebno_db_range, ber_vals, 'o-', label='Simulated')
    ber_th = [ber_theory_mqam(db,M) for db in ebno_db_range]
    plt.semilogy(ebno_db_range, ber_th, '--', label='Theory')
    plt.grid(True,which='both')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs Eb/N0, {M}-QAM OFDM (ofdm.py)')
    plt.legend()
    plt.show()

    # ---- Constellation Map (Received Symbols) ----
    plt.figure()
    plt.scatter(np.real(rx_syms[:2000]), np.imag(rx_syms[:2000]), s=5)
    plt.title(f'{M}-QAM Constellation Map (Received)')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # ---- PAPR CCDF ----
    papr_db = np.array(papr_vals)
    papr_range = np.linspace(0, np.max(papr_db), 200)
    ccdf = [np.mean(papr_db > x) for x in papr_range]
    papr_lin = 10 ** (papr_range / 20)
    ccdf_th = 1 - (1 - np.exp(-papr_lin**2)) ** N

    plt.figure()
    plt.semilogy(papr_range, ccdf, label='Simulated')
    plt.semilogy(papr_range, ccdf_th, '--', label='Theory')
    # plt.xlim((2, 13))
    plt.ylim((10 ** -6, 10 ** 0))
    plt.grid(True,which='both')
    plt.xlabel('PAPR (dB)')
    plt.ylabel('CCDF = Pr(PAPR > x)')
    plt.title('PAPR CCDF Curve (ofdm.py)')
    plt.legend()
    plt.show()

end = time.time()
# ~139s
print(f"\nTotal execution time: {end - start:.2f} seconds")
