import numpy as np

# ! calculates papr
def calculate_papr(tx_time):
    power = np.abs(tx_time) ** 2
    peak = np.max(power)
    avg = np.mean(power)
    if avg == 0: return 0.0
    return 10 * np.log10(peak / avg)

# Calculate the Cubic Metric (CM) for CCDF
def calculate_cm(signals):
    """
    Calculates the 3GPP standard Cubic Metric for a batch of OFDM symbols.
    """
    # 1. Calculate the RMS (Root Mean Square) voltage of each symbol
    rms_val = np.sqrt(np.mean(np.abs(signals) ** 2))

    # 2. Normalize the signal voltage
    # (Adding [:, None] allows us to divide the 2D matrix by the 1D RMS array)
    v_norm = np.abs(signals) / rms_val

    # 3. Calculate the RMS of the cubed normalized voltage (v_norm^3)^2 = v_norm^6
    rms_v_norm_cubed = np.sqrt(np.mean(v_norm ** 6))

    # 4. Convert the cubic RMS to decibels (dB)
    rcm = 20 * np.log10(rms_v_norm_cubed)

    # 5. Apply the standard 3GPP empirical constants
    rcm_ref = 1.52  # Reference Cubic Metric for a standard voice signal
    k_factor = 1.56  # Empirical slope factor for OFDM

    cm_values = (rcm - rcm_ref) / k_factor
    return cm_values
