import numpy as np

# Define 16-QAM mapping
qam16_mapping = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                          -1-3j, -1-1j, -1+3j, -1+1j,
                           3-3j,  3-1j,  3+3j,  3+1j,
                           1-3j,  1-1j,  1+3j,  1+1j])

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

def qpsk_mod(n_symbols):
    data = np.random.randint(0, 4, n_symbols)
    # Gray mapping: 00,01,11,10 -> 1+1j, 1-1j, -1-1j, -1+1j
    mapping = np.array([1+1j, 1-1j, -1-1j, -1+1j])
    symbols = mapping[data]
    # Normalize average power to 1 (each raw symbol has power 2)
    return symbols / np.sqrt(2)