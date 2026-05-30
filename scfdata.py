import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, scf, scf2
from ofdm.plots import plot_dataset_mapping
import torch
from nnscf import normalize
import os
import time

# todo: clean up this and other scf files

start = time.time()

# Parameters
N = 256                 # Number of Subcarriers
mid = N // 2
L = 4                   # Oversampling Factor
N_fft = N * L           # IFFT Size (extended to 1024)
# ! CP is only needed in ber
# CP = N // 4             # Cyclic Prefix
samples_per_L = 10000   # High value to capture the CCDF tail
cr_dB = 6
cr = 10 ** (cr_dB / 20)
iterations = 3

# modulation scheme
mod = "16qam"
# mod = "qpsk"
if mod == "16qam":
    M = 16
elif mod == "qpsk":
    M = 4
modulate, _ = get_modem(M)

# the actual size of the dataset
datasize = 100
# datasize = 120

params = f"SCF {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)"

# todo: see a way to add ber to the nnscf data so that we can use it in the loss function

pt_dir = "./pt_dir/"
os.makedirs(pt_dir, exist_ok=True)

datasize_minus_samples_per_L_loop = 0
samples_per_L_minus_scf_time = 0
scf_time = 0

# tx, rx = [], []
before_loop = time.time()
for i in range(datasize):
    loop_start = time.time() if i else before_loop
    tx_time, rx_time = [[], []], [[], []]
    for _ in range(samples_per_L):
        t1 = time.time()
        tx_data = np.random.randint(0, M, N)
        tx_symbols = modulate(tx_data)

        # Oversample and convert to time domain
        x_time = oversample_time(tx_symbols, N, L)

        # separately store real and imag parts (needed to generate data for NN model training)
        tx_time[0].append(np.real(x_time).astype(np.float32, copy=False))
        tx_time[1].append(np.imag(x_time).astype(np.float32, copy=False))

        t2 = time.time()
        samples_per_L_minus_scf_time += t2 - t1

        # Process SCF (1 Step replacing the 3 ICF iterations)
        x_scf, _, _ = scf2(x_time, cr, N, iterations=iterations)

        t3 = time.time()
        scf_time += t3 - t2

        # store the final iteration's real and imag parts separately
        rx_time[0].append(np.real(x_scf).astype(np.float32, copy=False))
        rx_time[1].append(np.imag(x_scf).astype(np.float32, copy=False))
        t4 = time.time()
        samples_per_L_minus_scf_time += t4 - t3
    t5 = time.time()

    tx_time = np.array(tx_time, dtype=np.float32)
    rx_time = np.array(rx_time, dtype=np.float32)

    # DIRECT IN-MEMORY NORMALIZATION & PYTORCH BUNDLING

    # Extract Real and Imaginary arrays and extract exact physical limits
    X_real_norm, (X_r_min, X_r_max) = normalize(tx_time[0])
    X_imag_norm, (X_i_min, X_i_max) = normalize(tx_time[1])
    Y_real_norm, (Y_r_min, Y_r_max) = normalize(rx_time[0])
    Y_imag_norm, (Y_i_min, Y_i_max) = normalize(rx_time[1])

    # Save Real File as PyTorch Dictionary
    torch.save({
        'X_norm': X_real_norm.detach().clone().to(torch.float32),
        'Y_norm': Y_real_norm.detach().clone().to(torch.float32),
        'X_min': X_r_min, 'X_max': X_r_max,
        'Y_min': Y_r_min, 'Y_max': Y_r_max
    }, os.path.join(pt_dir, f"{mod}_scf_part_{i:03d}_real.pt"))

    # Save Imaginary File as PyTorch Dictionary
    torch.save({
        'X_norm': X_imag_norm.detach().clone().to(torch.float32),
        'Y_norm': Y_imag_norm.detach().clone().to(torch.float32),
        'X_min': X_i_min, 'X_max': X_i_max,
        'Y_min': Y_i_min, 'Y_max': Y_i_max
    }, os.path.join(pt_dir, f"{mod}_scf_part_{i:03d}_imag.pt"))

    # Manually delete variables to free RAM for the next iteration
    del tx_time, rx_time, X_real_norm, X_imag_norm, Y_real_norm, Y_imag_norm
    # gc.collect()

    loop_end = time.time()
    print(f"Iteration {i + 1}/{datasize} completed in {loop_end - loop_start:.2f} seconds")
    print(f"{loop_end - before_loop:.2f} seconds passed since before loop start")
    t6 = time.time()
    datasize_minus_samples_per_L_loop += t6 - t5

title = f"Data Mapping\n{params}"
plot_dataset_mapping(os.path.join(pt_dir, f"{mod}_scf_part_{0:03d}_real.pt"), "NNSCF Real " + title)
plot_dataset_mapping(os.path.join(pt_dir, f"{mod}_scf_part_{0:03d}_imag.pt"), "NNSCF Imaginary " + title)

end = time.time()
# ~4:33.3 min
print(f'\nTotal execution time: {int((end - start) // 60)}:{(end - start) % 60:.1f} minutes')
print()
print(f"time taken in the scf calculations: {int(scf_time // 60)}:{scf_time % 60:.1f} minutes")
print(f"time taken in middle samples_per_L loop: {int(samples_per_L_minus_scf_time // 60)}:{samples_per_L_minus_scf_time % 60:.1f} minutes")
print(f"time taken in outer datasize loop: {int(datasize_minus_samples_per_L_loop // 60)}:{datasize_minus_samples_per_L_loop % 60:.1f} minutes")
print(f"time taken in plotting: {end - t6:.2f} seconds")
