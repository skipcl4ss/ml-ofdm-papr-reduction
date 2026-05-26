import numpy as np

# todo: update the docstring

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

def clip_time(tx_time, CR):
    """
    Clipping in time domain:
      A = CR * RMS(tx_time)
      y(n) = x(n) if |x| <= A
             A * exp(j*angle(x)) if |x| > A
    """
    mag = np.abs(tx_time)
    phase = np.angle(tx_time)
    rms = np.sqrt(np.mean(mag ** 2))
    A = CR * rms
    # ? why the phase
    clipped = np.where(mag <= A, tx_time, A * np.exp(1j * phase))
    return clipped, rms

def oversample_time(freq_symbols, N, L):
    # build oversampled frequency vector (zero padding in middle)
    mid = N // 2
    zeros = np.zeros((L - 1) * N)
    oversampled_freq = np.concatenate([freq_symbols[:mid], zeros, freq_symbols[mid:]], dtype=np.complex64)
    # time domain oversampled signal (original)
    tx_time_oversampled = np.fft.ifft(oversampled_freq) * L
    return tx_time_oversampled

def emulate_awgn_channel(tx_time, CP, SNR_dB, downsample=None):
    # Downsample back to original rate (take every L-th sample)
    if downsample != None:
        tx_time = tx_time[::downsample]

    tx_signal = np.concatenate([tx_time[-CP:], tx_time])
    rx_signal = awgn(tx_signal, SNR_dB)
    rx_ofdm = rx_signal[CP:]
    rx_symbols = np.fft.fft(rx_ofdm)
    return rx_symbols

def clip_and_filter_time(tx_time_oversampled, CR, N):
    """
    Steps:
      1) Create oversampled frequency vector by inserting zeros (L-1)*N in middle
      2) IFFT -> oversampled time
      3) Soft clip in time domain using CR
      4) FFT the clipped time -> keep only original subcarrier bins (i.e., set the inserted zeros bins to zero)
      5) IFFT back to oversampled time (filtered signal)
    Returns:
      tuple(original_oversampled_time, clipped_time_no_filter, clipped_filtered_time)
    """
    # soft clipping (no filtering)
    clipped_time, _ = clip_time(tx_time_oversampled, CR)
    clipped_filtered_time, _ = filter_time(clipped_time, N)

    return clipped_filtered_time, clipped_time

def filter_time(clipped_time, N):
    mid = N // 2
    # freq domain of clipped signal
    clipped_freq = np.fft.fft(clipped_time)
    # zero out the inserted bins (i.e., perform low-pass / band-limiting)
    # The original information lives in the positions we used earlier:
    kept_freq = np.zeros_like(clipped_freq)
    kept_freq[:mid] = clipped_freq[:mid]
    kept_freq[-mid:] = clipped_freq[-mid:]
    # IFFT to get clipped+filtered time signal
    clipped_filtered_time = np.fft.ifft(kept_freq)

    return clipped_filtered_time, kept_freq

def scf_time(tx_time_oversampled, CR, N, iterations=3):
    """
    Simplified Clipping and Filtering (SCF) algorithm.
    Approximates multiple ICF iterations with a single mathematical step.
    """
    # 0. Convert original signal to frequency domain (X_k)
    X_k = np.fft.fft(tx_time_oversampled)

    # 1. Soft clip in time domain and get clipping threshold A
    clipped_time, rms = clip_time(tx_time_oversampled, CR)
    A = CR * rms

    # 2. Calculate clipping noise f_n
    f_n = tx_time_oversampled - clipped_time

    # 3. Filter the clipping noise (F_tilde)
    _, F_k_tilde = filter_time(f_n, N)

    # 4. Calculate scaling factors alpha and beta
    # sigma is the standard deviation of the real/imag parts of the complex process
    sigma = rms / np.sqrt(2)

    # Alpha formula (Equation 10 mapped to Wang & Tellambura 2005)
    alpha = (2 * np.sqrt(2) / np.sqrt(3 * np.pi)) * (sigma / A)

    # Beta formula (Equations 10 & 11)
    num = 1 - (1 - alpha) ** (3 * iterations / 2)
    den = 1 - (1 - alpha) ** (3 / 2)
    beta = num / den

    # 5. Apply beta to the filtered noise and subtract from the original signal
    X_k_bar = X_k - beta * F_k_tilde

    # 6. Convert the final SCF signal back to the time domain
    x_n_bar = np.fft.ifft(X_k_bar)

    return x_n_bar, clipped_time

def scf_time2(tx_time_oversampled, CR, N, iterations=3):
    """
    Simplified Clipping and Filtering (SCF) algorithm.
    Approximates multiple ICF iterations with a single mathematical step.
    This function is extensively simplified compared to scf_time()
    """
    # 1. Soft clip in time domain
    clipped_time, _ = clip_time(tx_time_oversampled, CR)

    # 2. Calculate clipping noise (f_n)
    f_n = tx_time_oversampled - clipped_time

    # 3. Filter the clipping noise (F_tilde)
    # Reuses your existing filter_time function
    f_n_filtered, _ = filter_time(f_n, N)

    # # 4. Calculate scaling factors alpha and beta

    # Alpha formula (Equation 10 mapped to Wang & Tellambura 2005)
    alpha = 2 / (np.sqrt(3 * np.pi) * CR)

    # Beta formula (Equations 10 & 11)
    num = 1 - (1 - alpha) ** (3 * iterations / 2)
    den = 1 - (1 - alpha) ** (3 / 2)
    beta = num / den

    # 5. Apply beta to the filtered noise and subtract from the original signal
    x_n_bar = tx_time_oversampled - beta * f_n_filtered

    return x_n_bar, clipped_time