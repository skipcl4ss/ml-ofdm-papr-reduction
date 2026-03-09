import numpy as np

def qam16_mod(n_symbols):
    # Generate random integers 0-15
    data = np.random.randint(0, 16, n_symbols)
    # Define 16-QAM mapping
    mapping = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                        -1-3j, -1-1j, -1+3j, -1+1j,
                         3-3j,  3-1j,  3+3j,  3+1j,
                         1-3j,  1-1j,  1+3j,  1+1j])
    symbols = mapping[data]
    # Normalize power to 1 (Average power of this 16-QAM constellation is 10)
    return symbols / np.sqrt(10)

def qpsk_mod(n_symbols):
    data = np.random.randint(0, 4, n_symbols)
    # Gray mapping: 00,01,11,10 -> 1+1j, 1-1j, -1-1j, -1+1j
    mapping = np.array([1+1j, 1-1j, -1-1j, -1+1j])
    symbols = mapping[data]
    # Normalize average power to 1 (each raw symbol has power 2)
    return symbols / np.sqrt(2)