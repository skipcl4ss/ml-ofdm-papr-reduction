import numpy as np
from scipy import signal
from ofdm.papr import calculate_papr, calculate_cm
from ofdm.modem import qam16_mod
from ofdm.candf import clip_time
from ofdm.ccdf import plot_ccdf_compare
from nnicf import NNICFMapper, normalize
import time
import torch

start = time.time()

# Parameters
N = 256                 # Number of Subcarriers
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
CP = 32                 # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3

# IIR Low-Pass Filter design (Chebyshev Type I)
fp = 1 / L
b, a = signal.cheby1(N=4, rp=1, Wn=fp)

# Simulation
papr_unclipped = []
iterations_papr = [[] for _ in range(iterations)]
cm_unclipped = []
iterations_cm = [[] for _ in range(iterations)]

# todo: implement scf

tx_time = [[], []]
rx_time = [[], []]
for _ in range(samples_per_L):
    # todo: implement qpsk
    # Generate 16-QAM Symbols
    tx_data = np.random.randint(0, 16, N)
    symbols = qam16_mod(tx_data)

    # todo: use candf.py
    # Oversampling via Spectral Centering (Crucial for hitting 14dB)
    symbols_oversampled = np.zeros(N_fft, dtype=np.complex64)
    symbols_oversampled[:N // 2] = symbols[:N // 2]
    symbols_oversampled[-N // 2:] = symbols[N // 2:]

    # IFFT to Time Domain (Capturing true analog peaks)
    # Scale by L to maintain power through the zero-padded IFFT
    x_time = (np.fft.ifft(symbols_oversampled) * L).astype(np.complex64)

    # ! separately store real and imag parts (needed for the NN model)
    tx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
    tx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))

    # Capture Unclipped PAPR
    papr_unclipped.append(calculate_papr(x_time))
    cm_unclipped.append(calculate_cm(x_time))

    # Process C&F
    x_current = x_time
    for i in range(iterations):
        x_clipped = clip_time(x_current, cr)

        # Filtering: use lfilter (or filtfilt for zero-phase)
        x_current = signal.lfilter(b, a, x_clipped).astype(np.complex64)
        # x_current = signal.filtfilt(b, a, x_clipped)

        # Store PAPR of the current iterative result
        iterations_papr[i].append(calculate_papr(x_current))
        iterations_cm[i].append(calculate_cm(x_current))

        # ! store the final iteration's real and imag parts separately
        if i == iterations - 1:
            rx_time[0].append(np.real(x_current).astype(np.float32, copy=False))
            rx_time[1].append(np.imag(x_current).astype(np.float32, copy=False))
tx_time = np.array(tx_time, dtype=np.float32)
rx_time = np.array(rx_time, dtype=np.float32)

# * Compare with nnicf model

# 1. Normalize the data and turn it to torch tensors
tx_real, rx_real = normalize(tx_time, rx_time, "real")
tx_imag, rx_imag = normalize(tx_time, rx_time, "imag")

# 2. Load the trained models
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the empty models
NN_Mod_Re = NNICFMapper(N_fft).to(device)
NN_Mod_Im = NNICFMapper(N_fft).to(device)
# Inject the trained weights
NN_Mod_Re.load_state_dict(torch.load("./trained_models/mod_re_weights_80_20.pth", weights_only=True))
NN_Mod_Im.load_state_dict(torch.load("./trained_models/mod_im_weights_80_20.pth", weights_only=True))
NN_Mod_Re.eval()
NN_Mod_Im.eval()

# 3. Generate Predictions (No gradients needed for testing)
with torch.no_grad():
    predicted_real = NN_Mod_Re(tx_real).numpy()
    predicted_imag = NN_Mod_Im(tx_imag).numpy()

# 4. Reconstruct the Complex OFDM Signals
original_complex = tx_real.numpy() + 1j * tx_imag.numpy()
clipped_complex = rx_real.numpy() + 1j * rx_imag.numpy()
predicted_complex = predicted_real + 1j * predicted_imag

# 5. Calculate PAPR (Peak-to-Average Power Ratio) for CCDF
pred_papr, pred_cm = [], []
for i in range(len(original_complex)):
    pred_papr.append(calculate_papr(predicted_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))
pred_papr = np.array(pred_papr)
pred_cm = np.array(pred_cm)

# 6. Plot the CCDF
labels = ['Original OFDM', f'Clipped OFDM ({iterations} iterations)', 'NNICF Predicted OFDM']
plot_ccdf_compare([papr_unclipped, iterations_papr[-1], pred_papr], f"Original vs Clipped vs NNICF Predicted OFDM\n16QAM (N={N}, L={L}, CR={cr_dB}dB)\nModel 2 (80 Training 20 Testing)", labels)
plot_ccdf_compare([cm_unclipped, iterations_cm[-1], pred_cm], f"Original vs Clipped vs NNICF Predicted OFDM\n16QAM (N={N}, L={L}, CR={cr_dB}dB)\nModel 2 (80 Training 20 Testing)", labels, metric="CM")

end = time.time()
print(f"\nTotal execution time: {end - start:.2f} seconds")
