import numpy as np

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
    clipped = np.where(mag <= A, tx_time, A * np.exp(1j * phase)) # ? why
    return clipped

def clip_and_filter_ofdm(freq_symbols, N, L=4, CR=2.23):
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
    # build oversampled frequency vector (zero padding in middle)
    mid = N // 2
    zeros = np.zeros((L - 1) * N, dtype=complex)
    oversampled_freq = np.concatenate([freq_symbols[:mid], zeros, freq_symbols[mid:]])
    # time domain oversampled signal (original)
    tx_time_oversampled = np.fft.ifft(oversampled_freq)
    # soft clipping (no filtering)
    clipped_time = clip_time(tx_time_oversampled, CR)
    # freq domain of clipped signal
    clipped_freq = np.fft.fft(clipped_time)
    # zero out the inserted bins (i.e., perform low-pass / band-limiting)
    # The original information lives in the positions we used earlier:
    kept_freq = np.zeros_like(clipped_freq)
    kept_freq[:mid] = clipped_freq[:mid]
    kept_freq[-mid:] = clipped_freq[-mid:]
    # kept_freq[-(N-mid):] = clipped_freq[-(N-mid):]
    # IFFT to get clipped+filtered time signal
    clipped_filtered_time = np.fft.ifft(kept_freq)

    return clipped_filtered_time
    # return tx_time_oversampled, clipped_time, clipped_filtered_time