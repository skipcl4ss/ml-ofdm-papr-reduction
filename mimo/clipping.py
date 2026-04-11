import numpy as np

def clipping(x, CR, sigma=None):
    """
    Clip signal based on clipping ratio

    Parameters:
    -----------
    x : array_like
        Input signal
    CR : float
        Clipping Ratio
    sigma : float, optional
        Square root of variance of x. If not provided, it will be calculated.

    Returns:
    --------
    x_clipped : ndarray
        Clipped signal
    sigma : float
        Square root of variance (standard deviation)
    """
    x = np.asarray(x)

    if sigma is None:
        x_mean = np.mean(x)
        x_dev = x - x_mean
        sigma = np.sqrt(np.dot(x_dev, np.conj(x_dev)) / len(x))

    x_clipped = x.copy()
    CL = CR * sigma  # Clipping level

    # Find indices where absolute value exceeds clipping level
    ind = np.abs(x) > CL

    # Clip values: normalize and multiply by clipping level
    x_clipped[ind] = x[ind] / np.abs(x[ind]) * CL

    return x_clipped, sigma