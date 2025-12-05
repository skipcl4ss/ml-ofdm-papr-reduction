import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter

# # clipping.py

N = 2560
M = 16
L = 4
N_iter = 5
CR = 1.5
num_symbols = 10000
filter_order = 64

def make_filter():
    return firwin(filter_order + 1, 1 / L)

def qam_mod(bits):
    m = int(np.log2(M))
    gray = bits.reshape(-1, m)
    k = gray[:, 0] * 8 + gray[:, 1] * 4 + gray[:, 2] * 2 + gray[:, 3] * 1
    r = 2 * ((k % 4) - 1.5)
    i = 2 * ((k // 4) - 1.5)
    x = (r + 1j * i) / np.sqrt(10)
    return x

def ofdm_mod(symbols):
    symbols_oversampled = np.concatenate([symbols, np.zeros((L - 1) * N)])
    return np.fft.ifft(symbols_oversampled)

def papr(x):
    return 10 * np.log10(np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2))

def clip_signal(x, CR):
    A = CR * np.sqrt(np.mean(np.abs(x) ** 2))
    eps = 1e-12
    return np.where(np.abs(x) > A, A * x / (np.abs(x) + eps), x)

def filter_signal(x, h):
    return lfilter(h, 1, x)

def ccdf(values):
    values_sorted = np.sort(values)
    ccdf_vals = 1 - np.arange(len(values_sorted)) / len(values_sorted)
    return values_sorted, ccdf_vals

papr_original = []
papr_iter = []

h = make_filter()

for _ in range(num_symbols):
    bits = np.random.randint(0, 2, N * int(np.log2(M)))
    qam = qam_mod(bits)
    s = ofdm_mod(qam)

    papr_original.append(papr(s))

    x = s.copy()

    x = clip_signal(x, CR)
    x = filter_signal(x, h)
    papr_iter.append(papr(x))

orig_mean = np.mean(papr_original)
orig_median = np.median(papr_original)
orig_max = np.max(papr_original)

print("Original Mean PAPR (dB): ", orig_mean)
print("Original Median PAPR (dB): ", orig_median)
print("Original Max PAPR (dB): ", orig_max)

mean_papr = np.mean(papr_iter)
median_papr = np.median(papr_iter)
max_papr = np.max(papr_iter)

print("Mean PAPR after one iteration (dB): ", mean_papr)
print("Median PAPR after one iteration (dB): ", median_papr)
print("Max PAPR after one iteration (dB): ", max_papr)

iteration = np.arange(2)

plt.figure(figsize=(10, 6))
plt.plot(iteration, [orig_mean, mean_papr], 'o-', label="Mean PAPR")
plt.plot(iteration, [orig_median, median_papr], 's-', label="Median PAPR")
plt.plot(iteration, [orig_max, max_papr], 'd-', label="Max PAPR")

plt.axhline(orig_mean, color='gray', linestyle='--', label="Original Mean")
plt.axhline(orig_median, color='gray', linestyle='-.', label="Original Median")
plt.axhline(orig_max, color='gray', linestyle=':', label="Original Max")

plt.xlabel("Iteration Number")
plt.ylabel("PAPR (dB)")
plt.title("PAPR Statistics vs Clipping–Filtering Iterations")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

papr_o, ccdf_o = ccdf(papr_original)
plt.semilogy(papr_o, ccdf_o, label="Original")

pi, ci = ccdf(papr_iter)
plt.semilogy(pi, ci, label=f"Iteration 1")

plt.grid(True, which='both')
plt.xlabel("PAPR (dB)")
plt.ylabel("CCDF")
plt.title("PAPR CCDF – One Iteration Clipping & Filtering")
plt.legend()
plt.show()