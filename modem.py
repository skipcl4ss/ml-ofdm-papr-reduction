import numpy as np
from scipy.special import erfc

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