import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# 1. SYSTEM PARAMETERS
# ==========================================
# N_subcarriers = 512  # N = 512
N_subcarriers = 128
# L_oversampling = 4  # L factor mentioned in block diagram
L_oversampling = 8
# M_QAM = 16  # 16-QAM Modulation
M_QAM = 4
Bits_per_symbol = int(np.log2(M_QAM))  # 4 bits per symbol

# Simulation Iterations
# Note: The paper mentions 10,000 bits, but to reach a probability
# of 10^-4 for the CCDF, we need many OFDM symbols.
# We simulate enough symbols to get a smooth curve.
# N_ITERATIONS = 2000
N_ITERATIONS = 100

# Clipping Ratios (CR) to test
# CR_values = [3, 4, 5]
CR_values = [0.8, 1.0, 1.2, 1.4, 1.6]


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_16qam_constellation():
    """Generates normalized 16-QAM constellation points."""
    # 16-QAM: -3, -1, 1, 3 grid
    points = np.array([-3, -1, 1, 3])
    constellation = []
    for real in points:
        for imag in points:
            constellation.append(real + 1j * imag)
    constellation = np.array(constellation)
    # Normalize power to 1
    return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))


def ofdm_modulate(qam_symbols, N, L):
    """
    Performs IFFT with oversampling to generate OFDM signal.
    Paper mentions L.N Point IFFT.
    """
    # Create frequency domain array with zero padding for oversampling
    freq_data = np.zeros(N * L, dtype=complex)

    # Map subcarriers to inputs (maintain orthogonality)
    # Standard approach: split data into positive/negative frequencies
    freq_data[0:N // 2] = qam_symbols[0:N // 2]
    freq_data[-N // 2:] = qam_symbols[N // 2:]

    # IFFT
    time_signal = np.fft.ifft(freq_data) * np.sqrt(N * L)
    return time_signal


def calculate_papr_db(signal):
    """Calculates PAPR in dB: 10log10(P_peak / P_avg)."""
    power = np.abs(signal) ** 2
    peak_power = np.max(power)
    avg_power = np.mean(power)
    return 10 * np.log10(peak_power / avg_power)


def clip_and_filter(signal, cr, N, L):
    """
    Applies Clipping and Filtering technique.
    1. Clip amplitude > Threshold (A).
    2. Filter out-of-band noise (Peak regrowth control).
    """
    # --- CLIPPING STAGE ---
    # Calculate RMS (sigma)
    sigma = np.sqrt(np.mean(np.abs(signal) ** 2))

    # Calculate Threshold A = CR * sigma
    # Note: CR is typically linear here based on formula context,
    # but often defined in dB in plots. The paper implies A determined by CR.
    # We use linear CR ratio derived from the target dB reduction logic.
    # However, standard literature and the paper's formula A = CR * sigma
    # suggests CR is a voltage ratio here.

    # Convert CR from likely dB reference in text to linear voltage ratio if needed,
    # but strictly following formula A = CR * sigma:
    # We will treat the input CR as the voltage ratio factor.
    # However, usually CR=3,4,5 implies dB in PAPR context.
    # Let's approximate the paper's intent:
    # If CR=4 (linear) -> 20log10(4) = 12dB (Too high).
    # If CR is the target PAPR (dB), then A = sigma * 10^(CR_dB/20).
    # Looking at results: Original PAPR ~9.7dB. Result CR=4 is ~5.2dB.
    # This implies the CR variable is the Clipping Ratio in dB, or a specific factor.
    # Let's try standard definition: CR_linear = 10^(CR_dB/20).

    # Let's interpret CR values 3,4,5 as the parameter directly passed to formula.
    # A common interpretation in simulations for this paper's specific results:
    threshold = cr  # If CR is treated as a raw multiplier (rare)
    # OR Standard Definition:
    # A = sigma * 10^(CR_dB/20)

    # RE-EVALUATING PAPER CONTEXT:
    # Formula: A = CR * sigma.
    # If CR was 4, and sigma is 1, A=4. Max peaks are usually ~3-4 sigma.
    # So A=CR*sigma is likely correct linear interpretation.

    threshold = (10 ** (cr / 20)) * sigma  # Trying dB interpretation first (standard)
    # Actually, let's try the direct linear interpretation from eq
    # If CR=4, A = 4 * RMS. (This is very high, hardly any clipping).
    # Re-reading: "CR=4 ... produces PAPR value of 5.179 dB".
    # This suggests CR refers to the target Clipping Ratio in Volts normalized,
    # OR it's a specific index.
    # Let's stick to the most common implementation: Threshold based on CR in dB.
    threshold = (10 ** (cr / 20)) * sigma

    # Perform Clipping
    clipped_signal = signal.copy()
    mask = np.abs(clipped_signal) > threshold
    clipped_signal[mask] = threshold * np.exp(1j * np.angle(clipped_signal[mask]))

    # --- FILTERING STAGE ---
    # Remove out-of-band radiation
    # Convert to Freq Domain
    freq_data = np.fft.fft(clipped_signal)

    # Zero out the middle (oversampled) part (The stopband)
    # The data is at edges 0..N/2 and -N/2..end
    # The zeros should be in the center: N/2 to (L*N - N/2)
    freq_data[N // 2: int(L * N - N / 2)] = 0

    # Convert back to Time Domain
    filtered_signal = np.fft.ifft(freq_data)

    # return filtered_signal
    return clipped_signal, filtered_signal


# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
print(f"Running Simulation with {N_ITERATIONS} iterations...")
print(f"Targeting paper results at 10^-4 probability")

papr_original = []
papr_results = {cr: [] for cr in CR_values}
papr_unfiltered = {cr: [] for cr in CR_values}

constellation = get_16qam_constellation()

for i in range(N_ITERATIONS):
    # 1. Generate Random Data
    random_indices = np.random.randint(0, M_QAM, N_subcarriers)
    qam_data = constellation[random_indices]

    # 2. OFDM Modulation (IFFT)
    tx_signal = ofdm_modulate(qam_data, N_subcarriers, L_oversampling)

    # 3. Store Original PAPR
    papr_original.append(calculate_papr_db(tx_signal))

    # 4. Apply Clipping & Filtering for different CRs
    for cr in CR_values:
        # Note: The paper treats CR as a parameter.
        # We assume CR input is in dB for the threshold calculation
        # to match the reduction magnitude observed in results.
        # Based on results (CR=3 resulting in 3.2dB), it seems CR represents
        # the target clipping level in dB directly.

        # processed_signal = clip_and_filter(tx_signal, cr, N_subcarriers, L_oversampling)
        [processed_signal_unfiltered, processed_signal] = clip_and_filter(tx_signal, cr, N_subcarriers, L_oversampling)
        papr_results[cr].append(calculate_papr_db(processed_signal))
        papr_unfiltered[cr].append(calculate_papr_db(processed_signal_unfiltered))


# ==========================================
# 4. DATA PROCESSING (CCDF)
# ==========================================
def compute_ccdf(papr_array):
    x = np.sort(papr_array)
    y = 1 - np.arange(1, len(x) + 1) / len(x)
    return x, y


x_orig, y_orig = compute_ccdf(papr_original)

# ==========================================
# 5. VISUALIZATION
# ==========================================

# # --- Plot 1: Time Domain Signal (Replicating Fig 3 & 4) ---
# # We take the last iteration's signal for visualization
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# t = np.arange(len(tx_signal))
# plt.title('Time Domain OFDM Signal (Original vs Clipped)')
# plt.plot(t, np.abs(tx_signal), 'b', label='Original OFDM')
# # To match Fig 3, we show a zoomed or clipped version
# # Applying CR=4 for visualization
# # demo_clip = clip_and_filter(tx_signal, 4, N_subcarriers, L_oversampling)
# [demo_clip_unfiltered, demo_clip] = clip_and_filter(tx_signal, 4, N_subcarriers, L_oversampling)
# plt.grid(True)
# plt.ylabel('Amplitude')
# plt.ylim(0, np.max(np.abs(tx_signal)) * 1.1)
# plt.subplot(3, 1, 2)
# plt.plot(t, np.abs(demo_clip_unfiltered), 'g-', linewidth=1, label='Clipped (CR=4)')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.ylim(0, np.max(np.abs(tx_signal)) * 1.1)
# plt.axhline(np.max(demo_clip_unfiltered), color='gray', linestyle='--')
# plt.subplot(3, 1, 3)
# plt.plot(t, np.abs(demo_clip), 'r--', linewidth=1, label='Clipped (CR=4)')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.ylim(0, np.max(np.abs(tx_signal)) * 1.1)
# plt.axhline(np.max(demo_clip_unfiltered), color='gray', linestyle='--')
# plt.show()
# --- Plot 2: CCDF Curves (Replicating Fig 5) ---
plt.figure(figsize=(10, 7))

# Original
plt.semilogy(x_orig, y_orig, 'k-', linewidth=2, label='Original OFDM')

# Processed CRs
# colors = {3: 'r', 4: 'g', 5: 'b'}  # Colors from paper description
colors = {0.8: 'r', 1.0: 'g', 1.2: 'b', 1.4: 'm', 1.6: 'c'}
for cr in CR_values:
    x_cr, y_cr = compute_ccdf(papr_results[cr])
    label_text = f'CR={cr}'
    plt.semilogy(x_cr, y_cr, f'{colors[cr]}--o', markersize=3, label=label_text)
# colors = {3: 'm', 4: 'y', 5: 'c'}  # Colors from paper description
# colors = {0.7: 'y'}
for cr in CR_values:
    x_cr, y_cr = compute_ccdf(papr_unfiltered[cr])
    label_text = f'CR={cr} unfiltered'
    plt.semilogy(x_cr, y_cr, f'{colors[cr]}-^', markersize=5, label=label_text)

# Formatting to match Fig 5
plt.ylim(1e-2, 1)
# plt.xlim((0, 13))
plt.title('CCDF of PAPR (Clipping Filtering Technique) (iarjset.py)')
plt.xlabel('PAPR0 (dB)')
plt.ylabel('CCDF (Pr[PAPR > PAPR0])')
plt.grid(True, which="both", ls="-")
plt.legend()

# ==========================================
# 6. PRINT RESULTS COMPARISON
# ==========================================
print("\n--- RESULTS COMPARISON (at 10^-4 probability) ---")
print("Note: Values vary slightly due to random noise generation.")


# python
def get_papr_at_prob(x, y, target_prob=1e-4):
    import numpy as _np
    from scipy.interpolate import interp1d as _interp1d

    x = _np.asarray(x)
    y = _np.asarray(y)

    # Keep only positive probabilities to avoid log10(0)
    mask = y > 0
    if not _np.any(mask):
        return float(x[-1])

    x_valid = x[mask]
    y_valid = y[mask]

    # Work in log10-probability domain and ensure the x-axis for interp is increasing
    logy = _np.log10(y_valid)

    # If logy is decreasing (typical since CCDF decreases with x), reverse arrays
    if logy[0] > logy[-1]:
        logy = logy[::-1]
        x_valid = x_valid[::-1]

    try:
        f = _interp1d(logy, x_valid, kind='linear', fill_value='extrapolate', assume_sorted=True)
        return float(f(_np.log10(target_prob)))
    except Exception:
        return float(x_valid[-1])


val_orig = get_papr_at_prob(x_orig, y_orig)
print(f"Original OFDM PAPR (Target ~9.727 dB): {val_orig:.3f} dB")

for cr in CR_values:
    val_cr = get_papr_at_prob(*compute_ccdf(papr_results[cr]))
    print(f"CR={cr} PAPR (Paper Result):")
    if cr == 5: print(f"  -> Simulated: {val_cr:.3f} dB | Paper: ~7.003 dB")
    if cr == 4: print(f"  -> Simulated: {val_cr:.3f} dB | Paper: ~5.179 dB")
    if cr == 3: print(f"  -> Simulated: {val_cr:.3f} dB | Paper: ~3.279 dB")

plt.show()