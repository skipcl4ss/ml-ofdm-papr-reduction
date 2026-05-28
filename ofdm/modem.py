import numpy as np

# todo: see test.py and look for a way to generalize these functions

# Define 16-QAM and QPSK mappings
qam16_mapping  =  np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                            -1-3j, -1-1j, -1+3j, -1+1j,
                            +3-3j, +3-1j, +3+3j, +3+1j,
                            +1-3j, +1-1j, +1+3j, +1+1j])

def qam16_mod(data, bits=False):
    # # for if i ever want to embed the generation part
    # data = np.random.randint(0, 2, n_bits)
    # data = np.random.randint(0, 16, n_symbols)
    if bits:
        # weights = 1 << np.arange(int(np.log2(M)) - 1, -1, -1)
        groups = data.reshape(-1, int(np.log2(16)))  # first log2(16) bits per group
        weights = 1 << np.arange(int(np.log2(16)) - 1, -1, -1)
        data = groups @ weights
    symbols = qam16_mapping[data]
    # Normalize power to 1 (Average power of this 16-QAM constellation is 10)
    return symbols / np.sqrt(10)

def qam16_demod(rx_symbols, bits=False):
    """Takes noisy 16-QAM symbols and returns the most likely integers (0-15)."""
    # 1. Un-normalize the received symbols back to the original grid
    rx_scaled = rx_symbols * np.sqrt(10)

    # 2. Calculate the distance from each received symbol to all 16 ideal points
    # rx_scaled[:, None] turns it into a column, subtracting the row of MAPPING
    distances = np.abs(rx_scaled[:, None] - qam16_mapping[None, :])

    # 3. Find the index (which corresponds to the integer 0-15) of the minimum distance
    rx_data = np.argmin(distances, axis=1)

    if bits:
        # rx_data = np.unpackbits(rx_data.astype(np.uint8)[:, None], axis=1)[:, -int(np.log2(M)):].flatten()
        rx_data = np.unpackbits(rx_data.astype(np.uint8)[:, None], axis=1)[:, -int(np.log2(16)):].flatten()

    return rx_data

# ! this array should be more accurate, each two consecutive values work as intended in terms of gray coding when used in the function
# Gray mapping: 00,01,10,11 -> 1+1j, -1+1j, 1-1j, -1-1j
qpsk_mapping  =   np.array([+1+1j, -1+1j,
                            +1-1j, -1-1j])

def qpsk_mod(data, bits=False):
    # # for if i ever want to embed the generation part
    # data = np.random.randint(0, 2, n_bits)
    # data = np.random.randint(0, 4, n_symbols)
    if bits:
        # weights = 1 << np.arange(int(np.log2(M)) - 1, -1, -1)
        groups = data.reshape(-1, int(np.log2(4)))  # first log2(4) bits per group
        weights = 1 << np.arange(int(np.log2(4)) - 1, -1, -1)
        data = groups @ weights
    symbols = qpsk_mapping[data]
    # Normalize average power to 1 (each raw symbol has power 2)
    return symbols / np.sqrt(2)

def qpsk_demod(rx_symbols, bits=False):
    """Takes noisy QPSK symbols and returns the most likely integers (0-3)."""
    # 1. Un-normalize the received symbols back to the original grid
    rx_scaled = rx_symbols * np.sqrt(2)

    # 2. Calculate the distance from each received symbol to all 4 ideal points
    # rx_scaled[:, None] turns it into a column, subtracting the row of MAPPING
    distances = np.abs(rx_scaled[:, None] - qpsk_mapping[None, :])

    # 3. Find the index (which corresponds to the integer 0-3) of the minimum distance
    rx_data = np.argmin(distances, axis=1)

    if bits:
        # rx_data = np.unpackbits(rx_data.astype(np.uint8)[:, None], axis=1)[:, -int(np.log2(M)):].flatten()
        rx_data = np.unpackbits(rx_data.astype(np.uint8)[:, None], axis=1)[:, -int(np.log2(4)):].flatten()

    return rx_data

def get_modem(M):
    if M == 4:
        return qpsk_mod, qpsk_demod
    elif M == 16:
        return qam16_mod, qam16_demod
