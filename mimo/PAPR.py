import numpy as np

def PAPR(x):
    """
    Calculate Peak-to-Average Power Ratio (PAPR)

    Parameters:
    -----------
    x : array_like
        Input signal (can be complex)

    Returns:
    --------
    PAPR_dB : float
        PAPR in dB
    AvgP_dB : float
        Average power in dB
    PeakP_dB : float
        Peak (maximum) power in dB
    """
    x = np.asarray(x)
    Nx = len(x)

    # Extract real and imaginary parts
    xI = np.real(x)
    xQ = np.imag(x)

    # Calculate instantaneous power
    Power = xI ** 2 + xQ ** 2

    # Average power
    AvgP = np.sum(Power) / Nx
    AvgP_dB = 10 * np.log10(AvgP)

    # Peak power
    PeakP = np.max(Power)
    PeakP_dB = 10 * np.log10(PeakP)

    # PAPR
    PAPR_dB = 10 * np.log10(PeakP / AvgP)

    return PAPR_dB, AvgP_dB, PeakP_dB