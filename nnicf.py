import torch
import torch.nn as nn

# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(X_raw, Y_raw, part):
    # 1. Extract Real or Imaginary
    if part == "real":
        part_idx = 0
    elif part == "imag":
        part_idx = 1

    X_batch = torch.tensor(X_raw[part_idx, :, :], dtype=torch.float32)
    Y_batch = torch.tensor(Y_raw[part_idx, :, :], dtype=torch.float32)

    # 2. Normalize once and for all
    X_min, X_max = X_batch.min(), X_batch.max()
    Y_min, Y_max = Y_batch.min(), Y_batch.max()

    X_norm = 2.0 * ((X_batch - X_min) / (X_max - X_min)) - 1.0 if X_max != X_min else X_batch
    Y_norm = 2.0 * ((Y_batch - Y_min) / (Y_max - Y_min)) - 1.0 if Y_max != Y_min else Y_batch

    # 3. Return as a hyper-fast PyTorch file
    return X_norm, Y_norm

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
