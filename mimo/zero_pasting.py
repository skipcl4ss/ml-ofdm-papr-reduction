import numpy as np

def zero_pasting(x):
    """Paste zeros at the center half of the input sequence x"""
    N = len(x)
    M = int(np.ceil(N / 4))
    y = np.concatenate([x[:M], np.zeros(N // 2), x[N - M:]])
    return y