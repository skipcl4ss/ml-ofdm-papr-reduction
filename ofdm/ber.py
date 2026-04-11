import numpy as np
from scipy.special import erfc

def ber_theoretical(EbNo_dB, M):
    EbNo_lin = 10 ** (EbNo_dB / 10)

    if M == 2:
        return 0.5 * erfc(np.sqrt(EbNo_lin))
    elif M == 4:
        return 0.5 * erfc(np.sqrt(EbNo_lin))
    elif M == 8:
        k = 3
        return (1/k) * erfc(np.sqrt(k * EbNo_lin) * np.sin(np.pi/M))
    elif M == 16:
        return (3/8) * erfc(np.sqrt(0.4 * EbNo_lin))
    elif M == 64:
        return (7/24) * erfc(np.sqrt((1/7) * EbNo_lin))
    return 0.0