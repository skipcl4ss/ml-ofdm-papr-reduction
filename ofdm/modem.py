import numpy as np

# Define 16-QAM and QPSK mappings
qam16_mapping  =  np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                            -1-3j, -1-1j, -1+3j, -1+1j,
                            +3-3j,  3-1j,  3+3j,  3+1j,
                            +1-3j,  1-1j,  1+3j,  1+1j])

# Gray mapping: 00,01,11,10 -> 1+1j, 1-1j, -1-1j, -1+1j
qpsk_mapping  =   np.array([+1+1j,  1-1j,
                            -1-1j, -1+1j])

def qam16_mod(data):
    """Takes noisy 16-QAM symbols and returns the most likely integers (0-15)."""
    # data = np.random.randint(0, 16, n_symbols)
    symbols = qam16_mapping[data]
    # Normalize power to 1 (Average power of this 16-QAM constellation is 10)
    return symbols / np.sqrt(10)

def qam16_demod(rx_symbols):
    """Takes noisy 16-QAM symbols and returns the most likely integers (0-15)."""
    # 1. Un-normalize the received symbols back to the original grid
    rx_scaled = rx_symbols * np.sqrt(10)

    # 2. Calculate the distance from each received symbol to all 16 ideal points
    # rx_scaled[:, None] turns it into a column, subtracting the row of MAPPING
    distances = np.abs(rx_scaled[:, None] - qam16_mapping[None, :])

    # 3. Find the index (which corresponds to the integer 0-15) of the minimum distance
    rx_data = np.argmin(distances, axis=1)

    return rx_data

def qpsk_mod(data):
    # data = np.random.randint(0, 4, n_symbols)
    symbols = qpsk_mapping[data]
    # Normalize average power to 1 (each raw symbol has power 2)
    return symbols / np.sqrt(2)

# todo: needs revision, as i just copied the qam16_demod and changed accordingly
def qpsk_demod(rx_symbols):
    """Takes noisy 16-QPSK symbols and returns the most likely integers (0-3)."""
    # 1. Un-normalize the received symbols back to the original grid
    rx_scaled = rx_symbols * np.sqrt(2)

    # 2. Calculate the distance from each received symbol to all 4 ideal points
    # rx_scaled[:, None] turns it into a column, subtracting the row of MAPPING
    distances = np.abs(rx_scaled[:, None] - qpsk_mapping[None, :])

    # 3. Find the index (which corresponds to the integer 0-3) of the minimum distance
    rx_data = np.argmin(distances, axis=1)

    return rx_data

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

    def demodulate(self, symbols):
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
