import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

start = time.time()

# --- Configuration Based on Paper  ---
# OVERSAMPLING_FACTOR = 4  # L
# MODULATION_ORDER = 16  # 16-QAM
OVERSAMPLING_FACTOR = 8  # L
MODULATION_ORDER = 4  # 16-QAM
# "Clipping level 0.70" in the paper results in a PAPR drop from ~11.5 to ~8.9 dB.
# This implies a clipping threshold roughly 0.7 times the max peak, or a specific CR relative to RMS.
# We interpret this as a Clipping Ratio (CR) relative to RMS for stability.
# A CR of approx 1.4 - 1.6 usually achieves this reduction level.
# CLIPPING_RATIO = 0.7
CLIPPING_RATIO = [0.8, 1.0, 1.2, 1.4, 1.6]
# N_VALUES = [64, 128, 256, 512, 1024]
N_VALUES = [16, 32, 64, 128, 256, 512, 1024]
# NUM_SYMBOLS = 10000  # Number of OFDM symbols to simulate for CCDF smoothness
NUM_SYMBOLS = 100


def mapping_16qam(data_bits):
    """Maps bits to 16-QAM constellation points."""
    # Simplified mapping for simulation statistics (amplitude distribution matches)
    real = 2 * (data_bits[:len(data_bits) // 2] % 4) - 3
    imag = 2 * (data_bits[len(data_bits) // 2:] % 4) - 3
    return (real + 1j * imag) / np.sqrt(10)  # Normalize to unit power


def generate_ofdm_symbol(N, L):
    """Generates an oversampled OFDM symbol."""
    # 1. Generate random bits and map to QAM
    bits = np.random.randint(0, 4, N * 2)  # 4 levels per component for 16-QAM
    qam_symbols = mapping_16qam(bits)

    # 2. Oversampling via IFFT (Zero padding in frequency domain)
    # Create an array of size L*N
    freq_data = np.zeros(N * L, dtype=complex)

    # Insert data into subcarriers (Standard OFDM structure: DC in middle or edge)
    # Here we map to [1...N/2] and [L*N - N/2 + 1 ... L*N] to maintain symmetry
    freq_data[1:N // 2 + 1] = qam_symbols[N // 2:]
    freq_data[-N // 2:] = qam_symbols[:N // 2]

    # 3. IFFT to get time domain
    time_signal = np.fft.ifft(freq_data) * np.sqrt(N * L)  # Scaling
    return time_signal


def calculate_papr_db(signal):
    """Calculates PAPR in dB for a given signal vector."""
    power = np.abs(signal) ** 2
    peak_power = np.max(power)
    avg_power = np.mean(power)
    if avg_power == 0: return 0
    return 10 * np.log10(peak_power / avg_power)


def clip_signal(signal, cr):
    """
    Clips the signal at a threshold defined by Clipping Ratio (CR) * RMS.
    The paper mentions 'Clipping level 0.70'.
    """
    rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    threshold = cr * rms

    # Clipping formula: x_c = x * (threshold / |x|) if |x| > threshold
    amplitude = np.abs(signal)
    clipped_signal = np.copy(signal)
    mask = amplitude > threshold
    clipped_signal[mask] = signal[mask] / amplitude[mask] * threshold

    return clipped_signal


def filter_signal(signal, N, L):
    """
    Filters the signal to remove out-of-band distortion caused by clipping.
    Method: FFT -> Zero out non-data subcarriers -> IFFT.
    """
    # 1. FFT
    freq_domain = np.fft.fft(signal)

    # 2. Zero out out-of-band components (Simple brick-wall filter)
    # Keep only the original N subcarrier locations
    # (Indices: 0 to N/2 and L*N - N/2 to L*N)
    filtered_freq = np.zeros_like(freq_domain)
    filtered_freq[1:N // 2 + 1] = freq_domain[1:N // 2 + 1]
    filtered_freq[-N // 2:] = freq_domain[-N // 2:]

    # 3. IFFT
    filtered_signal = np.fft.ifft(filtered_freq)
    return filtered_signal


def run_simulation(N_vals):
    results = {}

    for N in N_vals:
        print(f"Simulating N={N}...")
        results[N] = {}

        for cr in CLIPPING_RATIO:
            papr_original = []
            papr_clipped = []
            papr_filtered = []

            example_signals = {}

            for i in range(NUM_SYMBOLS):
                sig_orig = generate_ofdm_symbol(N, OVERSAMPLING_FACTOR)
                sig_clip = clip_signal(sig_orig, cr)
                sig_filt = filter_signal(sig_clip, N, OVERSAMPLING_FACTOR)

                papr_original.append(calculate_papr_db(sig_orig))
                papr_clipped.append(calculate_papr_db(sig_clip))
                papr_filtered.append(calculate_papr_db(sig_filt))

                if i == NUM_SYMBOLS - 1:
                    example_signals['original'] = sig_orig
                    example_signals['clipped'] = sig_clip
                    example_signals['filtered'] = sig_filt

            results[N][cr] = {
                'papr_orig': np.array(papr_original),
                'papr_clip': np.array(papr_clipped),
                'papr_filt': np.array(papr_filtered),
                'signals': example_signals
            }

    return results


def plot_ccdf(results, N, cr=None):
    """
    If cr is None: plot original once and clipped/filtered curves for all CRs.
    If cr is provided: plot only that CR (plus original).
    """
    data_per_cr = results[N]
    cr_keys = sorted(data_per_cr.keys())

    plt.figure(figsize=(10, 7))

    def get_ccdf(papr_data):
        x = np.sort(papr_data)
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        return x, y

    # Use original from the first CR entry
    first_cr = cr_keys[0]
    x_orig, y_orig = get_ccdf(data_per_cr[first_cr]['papr_orig'])
    plt.semilogy(x_orig, y_orig, label='Original OFDM', color='black', linewidth=2)

    to_plot = [cr] if cr is not None else cr_keys
    colors = plt.cm.viridis(np.linspace(0, 1, len(to_plot)))

    for color, crv in zip(colors, to_plot):
        d = data_per_cr[crv]
        x_clip, y_clip = get_ccdf(d['papr_clip'])
        x_filt, y_filt = get_ccdf(d['papr_filt'])
        plt.semilogy(x_clip, y_clip, label=f'Clipped CR={crv}', color=color, linestyle='--')
        plt.semilogy(x_filt, y_filt, label=f'Clipped+Filt CR={crv}', color=color, linestyle='-')

    plt.title(f'CCDF of PAPR (N={N}) (ijcnis.py)')
    plt.xlabel('PAPR0 (dB)')
    plt.ylabel('CCDF P(PAPR > PAPR0)')
    plt.ylim([10 ** -2, 10 ** 0])
    plt.xlim(0, 13)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()


def plot_signals(results, N, cr=None):
    """
    Plot example signals for a given N and CR. If cr is None, pick the first CR.
    """
    data_per_cr = results[N]
    cr_keys = sorted(data_per_cr.keys())
    chosen_cr = cr if cr is not None else cr_keys[0]
    sigs = data_per_cr[chosen_cr]['signals']

    s_orig = np.abs(sigs['original'])
    s_clip = np.abs(sigs['clipped'])
    s_filt = np.abs(sigs['filtered'])

    x_axis = np.arange(len(s_orig))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].stem(x_axis, s_orig, basefmt=" ", markerfmt=".", linefmt="C0-")
    axs[0].set_title(f'Normal OFDM Signal (N={N})')
    axs[0].set_ylim(0, np.max(s_orig) * 1.1)

    axs[1].stem(x_axis, s_clip, basefmt=" ", markerfmt=".", linefmt="C0-")
    axs[1].set_title(f'Clipped OFDM Signal (CR={chosen_cr})')
    axs[1].set_ylim(0, np.max(s_orig) * 1.1)
    axs[1].axhline(np.max(s_clip), color='gray', linestyle='--')

    axs[2].plot(x_axis, s_filt, color='C0')
    axs[2].fill_between(x_axis, s_filt, color='C0', alpha=0.3)
    axs[2].set_title('Clipped and Filtered OFDM Signal')
    axs[2].set_ylim(0, np.max(s_orig) * 1.1)
    axs[2].axhline(np.max(s_clip), color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

# --- Execution ---
simulation_data = run_simulation(N_VALUES)

for N in N_VALUES:
    # Plot CCDF for all CRs (omit `cr=` to plot all)
    plot_ccdf(simulation_data, N)

    # Optionally plot signals for a specific CR (example: first CR)
    plot_signals(simulation_data, N)  # or plot_signals(simulation_data, N, cr=1.2)

end = time.time()
# ~7s
print(f"\nTotal execution time: {end - start:.2f} seconds")
