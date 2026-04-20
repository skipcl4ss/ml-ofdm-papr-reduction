import numpy as np

def awgn(tx_signal, SNR_dB):
    """Adds AWGN to a complex signal given a target SNR in dB."""
    # Calculate average signal power
    signal_power = np.mean(np.abs(tx_signal)**2)
    # Calculate required noise power
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear

    # Generate complex Gaussian noise
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    return tx_signal + noise
