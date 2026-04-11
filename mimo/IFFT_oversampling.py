import numpy as np

def IFFT_oversampling(X, N, L=1):
    """
    Zero-padding and NL-point IFFT for oversampling
    Equivalent to N-point IFFT with interpolation using oversampling factor L

    Parameters:
    -----------
    X : array_like
        Input frequency domain signal (length N)
    N : int
        FFT size
    L : int, optional
        Oversampling factor (default: 1, no oversampling)

    Returns:
    --------
    xt : ndarray
        Time domain signal after oversampling
    time : ndarray
        Time vector
    """
    X = np.asarray(X).flatten()
    NL = N * L
    T = 1 / NL
    time = np.arange(0, 1, T)

    # Zero-padding: insert zeros in the middle of the spectrum
    X_padded = np.concatenate([
        X[:N // 2],
        np.zeros(NL - N),
        X[N // 2:]
    ])

    # NL-point IFFT with scaling by L
    xt = L * np.fft.ifft(X_padded, NL)

    return xt, time