import numpy as np

def mapper(b, N=None):
    """
    Generate PSK/QAM modulated symbols

    Parameters:
    -----------
    b : int
        Number of bits per symbol (modulation order = 2^b)
    N : int, optional
        If provided, generates N random modulated symbols
        Otherwise, generates constellation points for [0:2^b-1]

    Returns:
    --------
    modulated_symbols : ndarray
        Modulated symbols
    Mod : str
        Modulation type string
    """
    M = 2 ** b  # Modulation order

    if b == 1:
        # BPSK
        Mod = 'BPSK'
        A = 1
        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)
        # BPSK: 0 -> -1, 1 -> +1
        modulated_symbols = A * (2 * data - 1)

    elif b == 2:
        # QPSK with pi/4 offset
        Mod = 'QPSK'
        A = 1
        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)
        # QPSK with pi/4 phase offset
        angles = 2 * np.pi * data / M + np.pi / 4
        modulated_symbols = A * np.exp(1j * angles)

    else:
        # QAM
        Mod = f'{M}QAM'
        Es = 1  # Symbol energy
        A = np.sqrt(3 / (2 * (M - 1)) * Es)

        if N is not None:
            data = np.random.randint(0, M, N)
        else:
            data = np.arange(M)

        # Gray-coded QAM constellation
        k = int(np.sqrt(M))  # Square QAM
        I = 2 * (data % k) - k + 1  # In-phase component
        Q = 2 * (data // k) - k + 1  # Quadrature component
        modulated_symbols = A * (I + 1j * Q)

    return modulated_symbols, Mod