import numpy as np
from ofdm.modem import get_modem
from ofdm.candf import oversample_time, scf, scf2
from ofdm.plots import plot_dataset_mapping
import torch
from nnscf import normalize
import os
import time

# todo: see a way to add ber to the data so that we can use it in the loss function

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

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
if mod == "16qam":
    M = 16
elif mod == "qpsk":
    M = 4
modulate, _ = get_modem(M)

# Number of entries in the dataset
data_size = 100

params = f"SCF {mod.upper()} (N={N}, L={L}, CR={cr_dB}dB)"

raw_dir = "./raw dir/"
relev = "./relevant files/"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(relev, exist_ok=True)

data_size_minus_samples_per_L_loop = 0
samples_per_L_minus_scf_time = 0
scf_time = 0

raw_limits = {
    'X_r_min': [],
    'X_r_max': [],
    'Y_r_min': [],
    'Y_r_max': [],
    'X_i_min': [],
    'X_i_max': [],
    'Y_i_min': [],
    'Y_i_max': []
}

# tx, rx = [], []
before_loop = time.time()
for i in range(data_size):
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

    # Save Real File as PyTorch Dictionary
    # ! turn into tensors to avoid weights_only error
    torch.save({
        'X_raw': torch.tensor(tx_time[0], dtype=torch.float32),
        'Y_raw': torch.tensor(rx_time[0], dtype=torch.float32)
    }, os.path.join(raw_dir, f"{mod}_scf_part_{i:02d}_real.pt"))

    # Save Imaginary File as PyTorch Dictionary
    torch.save({
        'X_raw': torch.tensor(tx_time[1], dtype=torch.float32),
        'Y_raw': torch.tensor(rx_time[1], dtype=torch.float32)
    }, os.path.join(raw_dir, f"{mod}_scf_part_{i:02d}_imag.pt"))

    # Save raw data limits
    raw_limits['X_r_min'].append(tx_time[0].min())
    raw_limits['X_r_max'].append(tx_time[0].max())
    raw_limits['Y_r_min'].append(rx_time[0].min())
    raw_limits['Y_r_max'].append(rx_time[0].max())
    raw_limits['X_i_min'].append(tx_time[1].min())
    raw_limits['X_i_max'].append(tx_time[1].max())
    raw_limits['Y_i_min'].append(rx_time[1].min())
    raw_limits['Y_i_max'].append(rx_time[1].max())

    # Manually delete variables to free RAM for the next iteration
    del tx_time, rx_time
    # gc.collect()

    loop_end = time.time()
    print(f"Iteration {i + 1}/{data_size} completed in {loop_end - loop_start:.2f} seconds")
    print(f"{loop_end - before_loop:.2f} seconds passed since before loop start")
    t6 = time.time()
    data_size_minus_samples_per_L_loop += t6 - t5

torch.save(raw_limits, os.path.join(relev, f"{mod}_scf_{data_size}_raw_limits.pt"))

t7 = time.time()
data_size_minus_samples_per_L_loop += t7 - t6

real_path = f"{mod}_scf_part_{0:02d}_real"
imag_path = f"{mod}_scf_part_{0:02d}_imag"
plot_dataset_mapping(os.path.join(raw_dir, real_path + ".pt"), f"NNSCF Real\n{params} Batch #{0:02d}", raw=True, save=os.path.join(relev, real_path + ".png"))
plot_dataset_mapping(os.path.join(raw_dir, imag_path + ".pt"), f"NNSCF Imaginary\n{params} Batch #{0:02d}", raw=True, save=os.path.join(relev, imag_path + ".png"))

# x = torch.load(os.path.join(raw_dir, real_path + ".pt"), weights_only=True)
# X = normalize(x['X_raw'], (min(raw_limits['X_r_min']), max(raw_limits['X_r_max'])))
# Y = normalize(x['Y_raw'], (min(raw_limits['Y_r_min']), max(raw_limits['Y_r_max'])))
# torch.save({
#     "X_norm": X,
#     "Y_norm": Y
# }, os.path.join(relev, "test.pt"))
# plot_dataset_mapping(os.path.join(relev, "test.pt"), f"NNSCF Real\n{params}")

end = time.time()
print(f'\nTotal execution time: {int((end - start) // 60)}:{(end - start) % 60:05.2f} minutes')
print()
print(f"time taken in the scf calculations: {int(scf_time // 60)}:{scf_time % 60:05.2f} minutes")
print(f"time taken in middle samples_per_L loop: {int(samples_per_L_minus_scf_time // 60)}:{samples_per_L_minus_scf_time % 60:05.2f} minutes")
print(f"time taken in outer data_size loop: {int(data_size_minus_samples_per_L_loop // 60)}:{data_size_minus_samples_per_L_loop % 60:05.2f} minutes")
print(f"time taken in plotting:  {int((end - t7) // 60)}:{(end - t7) % 60:05.2f} minutes")
# print(f"time taken in plotting:  {int((end - t6) // 60)}:{(end - t6) % 60:05.2f} minutes")
#
# # Total execution time: 5:25.12 minutes
# #
# # time taken in the scf calculations: 2:30.16 minutes
# # time taken in middle samples_per_L loop: 1:14.55 minutes
# # time taken in outer data_size loop: 0:38.05 minutes
# # time taken in plotting:  1:01.93 minutes