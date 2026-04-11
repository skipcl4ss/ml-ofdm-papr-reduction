import numpy as np

def addcp(x, Ncp):
    """
    Add cyclic prefix to OFDM symbols

    Parameters:
    -----------
    x : array_like
        Input signal (can be 1D or 2D array)
        If 2D, each column is treated as one OFDM symbol
    Ncp : int
        Length of cyclic prefix (number of samples)

    Returns:
    --------
    y : ndarray
        Signal with cyclic prefix added
    """
    x = np.asarray(x)

    # Handle 1D array
    if x.ndim == 1:
        N = len(x)
        cp = x[-Ncp:]
        y = np.concatenate([cp, x])
    else:
        # 2D array (matrix)
        N, num_symbols = x.shape
        y = np.zeros((N + Ncp, num_symbols), dtype=x.dtype)

        for k in range(num_symbols):
            # Take last Ncp samples and prepend them
            cp = x[-Ncp:, k]
            y[:, k] = np.concatenate([cp, x[:, k]])

    return y