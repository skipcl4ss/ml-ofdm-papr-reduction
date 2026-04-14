import torch
import torch.nn as nn
import numpy as np

# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! instead of returning only real or imag part of two datasets (tx and rx), return both real and image of only one set
# ! works with data of shapes (2, samples_per_L, N * L), or (2, N * L) where 2 is for real and imag data
def normalize(X_raw, shape_len):
    if shape_len == 2:
        X_batch_real = torch.tensor(X_raw[0, :], dtype=torch.float32)
        X_batch_imag = torch.tensor(X_raw[1, :], dtype=torch.float32)
    elif shape_len == 3:
        X_batch_real = torch.tensor(X_raw[0, :, :], dtype=torch.float32)
        X_batch_imag = torch.tensor(X_raw[1, :, :], dtype=torch.float32)

    # Normalize once and for all
    X_min_real, X_max_real = X_batch_real.min(), X_batch_real.max()
    X_min_imag, X_max_imag = X_batch_imag.min(), X_batch_imag.max()

    X_norm_real = 2.0 * ((X_batch_real - X_min_real) / (X_max_real - X_min_real)) - 1.0 if X_max_real != X_min_real else X_batch_real
    X_norm_imag = 2.0 * ((X_batch_imag - X_min_imag) / (X_max_imag - X_min_imag)) - 1.0 if X_max_imag != X_min_imag else X_batch_imag

    # 3. Return as a hyper-fast PyTorch file
    return X_norm_real, X_norm_imag, (X_min_real, X_max_real), (X_min_imag, X_max_imag)

# ! should be used on real and imag separately
# fixed: no longer gives the exact same values as pred_norm, since it does not have access to the original data
def denormalize(pred_norm, minmax):
    raw_min, raw_max = minmax
    raw_min, raw_max = np.array([raw_min]), np.array([raw_max])
    pred_raw = ((pred_norm + 1.0) / 2.0) * (raw_max - raw_min) + raw_min if raw_max != raw_min else pred_norm
    return pred_raw

class TriangularActivation(nn.Module):
    """
    Implements the triangular activation function used in the paper's hidden layers.
    Mathematically equivalent to MATLAB's 'tribas': f(x) = max(1 - |x|, 0)
    """
    def forward(self, x):
        return torch.clamp(1.0 - torch.abs(x), min=0.0)

class NNICFMapper(nn.Module):
    def __init__(self, input_size):
        super(NNICFMapper, self).__init__()
        # First hidden layer: 2 neurons
        self.hidden1 = nn.Linear(input_size, 2)
        # Second hidden layer: 1 neuron
        self.hidden2 = nn.Linear(2, 1)
        # Output layer: Maps back to the original subcarrier points
        self.output = nn.Linear(1, input_size)
        # Custom triangular activation
        self.activation = TriangularActivation()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x) # Standard linear output
        return x
