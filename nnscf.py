import torch
import torch.nn as nn

# todo: maybe i could migrate most of the nnscf notebook dependencies here and use this script for imports

# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard Mean Squared Error loss
# todo: see how can i use loss function to reduce ber
criterion = nn.MSELoss()

# ! instead of returning two parts (real and imag) of two datasets (tx and rx), returns only one part of one dataset
# todo: fix and merge it with renormalization notebook
def normalize(X_raw):
    X_batch = torch.tensor(X_raw, dtype=torch.float32)

    # Convert PyTorch 0-D tensors to standard Python floats
    X_min, X_max = X_batch.min().item(), X_batch.max().item()

    X_norm = 2.0 * ((X_batch - X_min) / (X_max - X_min)) - 1.0 if X_max != X_min else X_batch

    # 3. Return as a hyper-fast PyTorch file
    return X_norm, (X_min, X_max)

# ! should be used on real and imag of each entry in the dataset separately
# ! to normalize the whole dataset using global values, run renormalization.ipynb after this function
def denormalize(pred_norm, minmax):
    raw_min, raw_max = minmax
    pred_raw = ((pred_norm + 1.0) / 2.0) * (raw_max - raw_min) + raw_min if raw_max != raw_min else pred_norm
    return pred_raw

class TriangularActivation(nn.Module):
    """
    Implements the triangular activation function used in the paper's hidden layers.
    Mathematically equivalent to MATLAB's 'tribas': f(x) = max(1 - |x|, 0)
    """
    def forward(self, x):
        return torch.clamp(1.0 - torch.abs(x), min=0.0)

class NNSCFMapper(nn.Module):
    def __init__(self):
        super(NNSCFMapper, self).__init__()
        # First hidden layer: 2 neurons
        self.hidden1 = nn.Linear(1, 2)
        # Second hidden layer: 1 neuron
        self.hidden2 = nn.Linear(2, 1)
        # Output layer: Maps back to the original subcarrier points
        self.output = nn.Linear(1, 1)
        # Custom triangular activation
        self.activation = TriangularActivation()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x) # Standard linear output
        return x
