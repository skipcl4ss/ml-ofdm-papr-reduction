import numpy as np
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.plots import plot_ccdf, plot_ccdf_compare
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from nnscf import NNSCFMapper, device, denormalize, criterion
import os
import time

# -----------------------------------------------------------------------------

# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
print(f"PyTorch using {torch.get_num_threads()} threads on {device}")

# -----------------------------------------------------------------------------
start = time.time()

samples_per_L = 10000

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
print(f"Using {mod.upper()} modulation technique")

# Clipping technique
# tech = "icf"
tech = "scf"
print(f"Using {tech} clipping technique")

# Hyperparameters
epochs = 100
# using 0.001 for Adam and 0.01 for LBFGS as default values
# lr = 0.001
lr = 0.1
lr_list = str(float(lr)).split(".")
lr_str = f"dot{lr_list[1]}" if lr < 1 else f"{lr_list[0]}dot{lr_list[1]}"
print(f"Hyperparameters: epochs = {epochs}, learning rate = {lr} ({lr_str} used in naming files), batch size = None (for now)")

# Optimizer
# opt = "Adam"
opt = "LBFGS"
print(f"Using {opt} optimizer")

# Data splitting
# ! make sure that the splitting is the same as in renormalization.ipynb
data_size = 100
train_size = 70
val_size = 10
test_size = data_size - train_size - val_size
if test_size:
    one_batch = None
else:
    test_size = 1
    one_batch = "00"
train_val_test_str = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"
print(f"dataset of size {data_size} is split into {train_size} training files, and {val_size} validation and {test_size} testing batches respectively ({train_val_test_str} used in naming files)")

batch_suffix = ""
if one_batch:
    batch_suffix = f"Batch #{one_batch}"
    print(f"This one batch is of index {one_batch}")
    if train_size + val_size == data_size:
        print(batch_suffix, "was already used in training")

params = f"{opt} optimizer {tech.upper()} {mod.upper()} lr = {lr}\n({train_size} Training/{val_size} Validation/{test_size} Testing){" (" + batch_suffix + ")" if batch_suffix else ''}"
print(params)

samples_per_L = 10000

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
print(f"Using {mod.upper()} modulation technique")

# clipping technique
# tech = "icf"
tech = "scf"
print(f"Using {tech} clipping technique")

# Hyperparameters
epochs = 100
# using 0.001 for Adam and 0.01 for LBFGS as default values
# lr = 0.001
lr = 0.1
lr_list = str(float(lr)).split(".")
lr_str = f"dot{lr_list[1]}" if lr < 1 else f"{lr_list[0]}dot{lr_list[1]}"
print(f"Hyperparameters: epochs = {epochs}, learning rate = {lr} ({lr_str} used in naming files), batch size = None (for now)")

# Optimizer
# opt = "Adam"
opt = "LBFGS"
print(f"Using {opt} optimizer")

# Data splitting
# ! make sure that the splitting is the same as in renormalization.ipynb
data_size = 100
train_size = 70
val_size = 10
test_size = data_size - train_size - val_size
if test_size:
    one_batch = None
else:
    test_size = 1
    one_batch = "00"
train_val_test_str = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"
print(f"dataset of size {data_size} is split into {train_size} training files, and {val_size} validation and {test_size} testing batches respectively ({train_val_test_str} used in naming files)")

batch_suffix = ""
if one_batch:
    batch_suffix = f"Batch #{one_batch}"
    print(f"This one batch is of index {one_batch}")
    if train_size + val_size == data_size:
        print(batch_suffix, "was already used in training")

params = f"{opt} optimizer {tech.upper()} {mod.upper()} lr = {lr}\n({train_size} Training/{val_size} Validation/{test_size} Testing){" (" + batch_suffix + ")" if batch_suffix else ''}"
print(params)

raw_dir = "./raw dir/"
norm_dir = "./norm dir/"
relev = "./relevant files/"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(norm_dir, exist_ok=True)
os.makedirs(relev, exist_ok=True)
print(raw_dir, "is the location of the raw unnormalized data files")
print(norm_dir, "is the location of the normalized data files")
print(relev, "is the location of other important files such as checkpoints and PNGs")

# -----------------------------------------------------------------------------
cell3 = time.time()

class FastOFDMDataset(Dataset):
    def __init__(self, pt_folder_path, part):
        self.folder_path = pt_folder_path
        self.part = part

    def __len__(self):
        return data_size # Assuming exactly data_size pre-computed .pt files, regardless of train/test split

    def __getitem__(self, idx):
        # Instantly loads the pre-normalized tensors directly into memory!
        file_path = os.path.join(self.folder_path, f"{mod}_{tech}_part_{idx:02d}_{self.part}.pt")
        # Load the dictionary
        data_pkg = torch.load(file_path, weights_only=False)

        # Return the tensors AND the scaling limits
        return (data_pkg['X_norm'], data_pkg['Y_norm'],
                data_pkg['X_min'], data_pkg['X_max'],
                data_pkg['Y_min'], data_pkg['Y_max'])

# -----------------------------------------------------------------------------
cell6 = time.time()

dataset_real = FastOFDMDataset(norm_dir, part='real')
dataset_imag = FastOFDMDataset(norm_dir, part='imag')

# -----------------------------------------------------------------------------
cell10 = time.time()

# ! this part is actually unnecessary in the notebook as the checkpoints are already loaded
checkpoint_re = torch.load(
    os.path.join(relev, f"{opt}_{mod}_{tech}_weights_limits_re_{train_val_test_str}_{lr_str}.pth"),
    weights_only=False
)
checkpoint_im = torch.load(
    os.path.join(relev, f"{opt}_{mod}_{tech}_weights_limits_im_{train_val_test_str}_{lr_str}.pth"),
    weights_only=False
)

# 1. Load the Saved Models
Test_Mod_Re = NNSCFMapper().to(device)
Test_Mod_Im = NNSCFMapper().to(device)

Test_Mod_Re.load_state_dict(checkpoint_re['model_state_dict'])
Test_Mod_Im.load_state_dict(checkpoint_im['model_state_dict'])

Test_Mod_Re.eval()
Test_Mod_Im.eval()

# 2. Grab the saved normalization limits
# X_r_min = checkpoint_re['X_min']
# X_r_max = checkpoint_re['X_max']
Y_r_min = checkpoint_re['Y_min']
Y_r_max = checkpoint_re['Y_max']

# X_i_min = checkpoint_im['X_min']
# X_i_max = checkpoint_im['X_max']
Y_i_min = checkpoint_im['Y_min']
Y_i_max = checkpoint_im['Y_max']

# -----------------------------------------------------------------------------

if one_batch:
    # 2. Grab one batch of data to test (Load the dictionary packages)
    pkg_real = torch.load(os.path.join(norm_dir, f"{mod}_{tech}_part_{one_batch}_real.pt"), weights_only=False)
    pkg_imag = torch.load(os.path.join(norm_dir, f"{mod}_{tech}_part_{one_batch}_imag.pt"), weights_only=False)
    # Extract tensors
    test_X_real, test_Y_real = pkg_real['X_norm'], pkg_real['Y_norm']
    test_X_imag, test_Y_imag = pkg_imag['X_norm'], pkg_imag['Y_norm']

    # 3. Generate Predictions (Reshape, Predict, Reshape back)
    with torch.no_grad():
        # Memoryless flattening
        X_real_flat = test_X_real.view(-1, 1).to(device)
        X_imag_flat = test_X_imag.view(-1, 1).to(device)

        predicted_real = Test_Mod_Re(X_real_flat).view_as(test_X_real).cpu().numpy()
        predicted_imag = Test_Mod_Im(X_imag_flat).view_as(test_X_imag).cpu().numpy()

    # Denormalize using exact dictionary limits
    # ? which limits should i use when using one_batch, local or global?
    # pred_denorm_real = denormalize(predicted_real, (pkg_real['Y_min'], pkg_real['Y_max']))
    # pred_denorm_imag = denormalize(predicted_imag, (pkg_imag['Y_min'], pkg_imag['Y_max']))
    pred_denorm_real = denormalize(predicted_real, (Y_r_min, Y_r_max))
    pred_denorm_imag = denormalize(predicted_imag, (Y_i_min, Y_i_max))

    # 4. Reconstruct the Complex OFDM Signals
    original_complex = test_X_real.numpy() + 1j * test_X_imag.numpy()
    clipped_complex = test_Y_real.numpy() + 1j * test_Y_imag.numpy()
    predicted_complex = pred_denorm_real + 1j * pred_denorm_imag
else:
    # 2. Setup the Test Data (Files (train_size + val_size) -> data_size - 1)
    test_dataset_real = Subset(dataset_real, range(train_size + val_size, data_size))
    test_dataset_imag = Subset(dataset_imag, range(train_size + val_size, data_size))

    # We set shuffle=False to ensure real and imag batches stay perfectly aligned
    test_loader_real = DataLoader(test_dataset_real, batch_size=None, shuffle=False)
    test_loader_imag = DataLoader(test_dataset_imag, batch_size=None, shuffle=False)

    # 3. Generate Predictions for all test_size test files
    all_original, all_clipped, all_predicted = [], [], []

    test_loss_real = 0.0
    test_loss_imag = 0.0
    with torch.no_grad():
        for (X_real, Y_real, _, _, _, _), (X_imag, Y_imag, _, _, _, _) in zip(test_loader_real, test_loader_imag):
            # Move inputs to device
            X_real, X_imag = X_real.to(device), X_imag.to(device)

            # 1. Flatten for memoryless prediction
            X_real_flat = X_real.view(-1, 1)
            X_imag_flat = X_imag.view(-1, 1)

            # Predict
            # ! shouldnt use .numpy() method because of criterion
            pred_real_flat = Test_Mod_Re(X_real_flat)
            pred_imag_flat = Test_Mod_Im(X_imag_flat)

            # 2. Calculate MSE Loss on the flat tensors
            loss_real = criterion(pred_real_flat, Y_real.view(-1, 1))
            loss_imag = criterion(pred_imag_flat, Y_imag.view(-1, 1))

            test_loss_real += loss_real.item()
            test_loss_imag += loss_imag.item()

            # 3. Reshape back to the original array shape (e.g. [10000, 1024])
            pred_real = pred_real_flat.view_as(X_real).cpu().numpy()
            pred_imag = pred_imag_flat.view_as(X_imag).cpu().numpy()

            # 4. Denormalize using the exact physical limits from the dictionary!
            pred_denorm_real = denormalize(pred_real, (Y_r_min, Y_r_max))
            pred_denorm_imag = denormalize(pred_imag, (Y_i_min, Y_i_max))

            # 5. Reconstruct complex signals
            orig_complex = X_real.cpu().numpy() + 1j * X_imag.cpu().numpy()
            clip_complex = Y_real.cpu().numpy() + 1j * Y_imag.cpu().numpy()
            pred_complex = pred_denorm_real + 1j * pred_denorm_imag

            all_original.append(orig_complex)
            all_clipped.append(clip_complex)
            all_predicted.append(pred_complex)

    # Print the final average test metrics
    avg_test_loss_real = test_loss_real / len(test_loader_real)
    avg_test_loss_imag = test_loss_imag / len(test_loader_imag)

    print(f'\n--- Test Set Metrics ({test_size} Unseen Files) ---')
    print(f'Average Real Module MSE Loss: {avg_test_loss_real:.6f}')
    print(f'Average Imag Module MSE Loss: {avg_test_loss_imag:.6f}')

    # 4. Concatenate the test_size batches into one massive array for evaluation
    original_complex = np.concatenate(all_original, axis=0)
    clipped_complex = np.concatenate(all_clipped, axis=0)
    predicted_complex = np.concatenate(all_predicted, axis=0)

    print(f'Testing complete! Aggregated {test_size * samples_per_L} OFDM symbols.')

# -----------------------------------------------------------------------------
cell11 = time.time()

# 5. Calculate PAPR and CM for CCDF
# orig_papr, clip_papr, pred_papr = [], [], []
orig_cm, clip_cm, pred_cm = [], [], []

# Iterate through the arrays
for i in range(test_size * samples_per_L if not one_batch else samples_per_L):
    # orig_papr.append(calculate_papr(original_complex[i]))
    # clip_papr.append(calculate_papr(clipped_complex[i]))
    # pred_papr.append(calculate_papr(predicted_complex[i]))

    orig_cm.append(calculate_cm(original_complex[i]))
    clip_cm.append(calculate_cm(clipped_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))

# orig_papr, clip_papr, pred_papr = np.array(orig_papr), np.array(clip_papr), np.array(pred_papr)
orig_cm, clip_cm, pred_cm = np.array(orig_cm), np.array(clip_cm), np.array(pred_cm)

# 6. Plot the CCDF
title = f'NN{tech.upper()} Predicted OFDM\n{params}'
labels = ['Original OFDM', f'{tech.upper()}', f'NN{tech.upper()} Predicted OFDM']
# papr_list = [orig_papr, clip_papr, pred_papr]
cm_list = [orig_cm, clip_cm, pred_cm]

# plot_ccdf_compare(papr_list, f'Original vs {tech.upper()} vs {title}', labels)
plot_ccdf_compare(cm_list, f'Original vs {tech.upper()} vs {title}', labels, metric="CM")

# fixme: both percentile and max are not the most efficient solutions

# todo: find a way to embed the floor part into the plotting function
# 1. Define the target y-levels (probabilities)
# papr_target_y = 1e-4
cm_target_y = 1e-3

# 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
# papr_percentile = (1.0 - papr_target_y) * 100.0
cm_percentile = (1.0 - cm_target_y) * 100.0

# 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
# (This completely replaces the need for y_axis, np.where, and manual sorting!)
# papr_vlines = [
#     np.percentile(orig_papr, papr_percentile),
#     np.percentile(clip_papr, papr_percentile),
#     np.percentile(pred_papr, papr_percentile)
# ]

cm_vlines = [
    np.percentile(orig_cm, cm_percentile),
    np.percentile(clip_cm, cm_percentile),
    np.percentile(pred_cm, cm_percentile)
]

# cm_vlines = [
#     np.max(orig_cm),
#     np.max(clip_cm),
#     np.max(pred_cm)
# ]

# # 4. Plot!
# # papr_image_path = os.path.join(relev, f"{opt}_{mod}_{train_val_test_str}_{lr_str}_papr.png")
# plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

# cm_image_path = os.path.join(relev, f"{opt}_{mod}_{tech}_{train_val_test_str}_{lr_str}_cm.png")
plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)

# -----------------------------------------------------------------------------
end = time.time()
# ~98s at test_size = 20
# ~6s at one_batch != None
print(f"\nTotal execution time: {end - start:.2f} seconds")
print(f"Imports and Device Setup (Cell 1) time: {cell3 - start:.2f} seconds")
print(f"Highly Optimized Dataset Class (Cell 3) time: {cell6 - cell3:.2f} seconds")
print(f"Initialization and DataLoaders (Cell 6) time: {cell10 - cell6:.2f} seconds")
print(f"Testing (Cell 10) time: {cell11 - cell10:.2f} seconds")
print(f"Plotting (Cell 11) time: {end - cell11:.2f} seconds")
