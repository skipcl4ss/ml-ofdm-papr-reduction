import numpy as np

# ! calculates papr
def calculate_papr(tx_time):
    power = np.abs(tx_time) ** 2
    peak = np.max(power)
    avg = np.mean(power)
    if avg == 0: return 0.0
    return 10 * np.log10(peak / avg)