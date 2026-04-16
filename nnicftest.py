import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import time
from ofdm.plots import plot_ccdf, plot_ccdf_compare
from ofdm.metrics import calculate_papr, calculate_cm
from nnicf import NNICFMapper, device, denormalize, criterion

# -----------------------------------------------------------------------------

start = time.time()
samples_per_L = 10000

# Modulation scheme
mod = "16qam"
# mod = "qpsk"
print(f"Using {mod.upper()} modulation technique")

# Hyperparameters
epochs = 100
lr = 0.001 # originally 0.001
lr_str = "dot" + str(lr).split(".")[1]
print(f"Hyperparameters: epochs = {epochs}, learning rate = {lr} ({lr_str} used in naming files), batch size = None (for now)")

# Data splitting
train_size = 80
test_size = (100 - train_size) or 1
train_test_str = f"{train_size}_{test_size}"
if train_size < 100:
    one_batch = None
elif train_size == 100:
    one_batch = "00"
print(f"dataset is split into {train_size} training files and {test_size} batches ({train_test_str} used in naming files, while {one_batch} is the index of used batch if testing on 1 batch)")

if train_size == 100:
    print("This 1 batch is of index 00, and was already used in training")

batch_suffix = f" (Batch #{one_batch})" if one_batch else ""
params = f"{mod.upper()} lr = {lr} ({train_size} Training/{test_size if not one_batch else 1} Testing){batch_suffix}"
print(params)

pt_dir = "./pt_dir/"
model_dir = "./new architecture/"
graph_dir = "./new architecture/"
os.makedirs(pt_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)
print(f'"{pt_dir}", "{model_dir}", and "{graph_dir}" are the 3 locations for pt files, nn model weights and graphs respectively')

# -----------------------------------------------------------------------------

# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
print(f"PyTorch using {torch.get_num_threads()} threads on {device}")

# -----------------------------------------------------------------------------

class FastOFDMDataset(Dataset):
    def __init__(self, pt_folder_path, part):
        self.folder_path = pt_folder_path
        self.part = part

    def __len__(self):
        return 100 # Assuming exactly 100 pre-computed .pt files, regardless of train/test split

    def __getitem__(self, idx):
        # Instantly loads the pre-normalized tensors directly into memory!
        file_path = os.path.join(self.folder_path, f"{mod}_tx_rx_32_part_{idx:02d}_{self.part}.pt")
        # Load the dictionary
        data_pkg = torch.load(file_path, weights_only=False)

        # Return the tensors AND the scaling limits
        return (data_pkg['X_norm'], data_pkg['Y_norm'],
                data_pkg['X_min'], data_pkg['X_max'],
                data_pkg['Y_min'], data_pkg['Y_max'])

# -----------------------------------------------------------------------------

# 1. Load the Saved Models
Test_Mod_Re = NNICFMapper().to(device)
Test_Mod_Im = NNICFMapper().to(device)

Test_Mod_Re.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_re_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))
Test_Mod_Im.load_state_dict(torch.load(os.path.join(model_dir, f"{mod}_mod_im_weights_{train_test_str}_{lr_str}.pth"), weights_only=True))

Test_Mod_Re.eval()
Test_Mod_Im.eval()

# -----------------------------------------------------------------------------

if one_batch:
    # 2. Grab one batch of data to test (Load the dictionary packages)
    pkg_real = torch.load(os.path.join(pt_dir, f"{mod}_tx_rx_32_part_{one_batch}_real.pt"), weights_only=False)
    pkg_imag = torch.load(os.path.join(pt_dir, f"{mod}_tx_rx_32_part_{one_batch}_imag.pt"), weights_only=False)
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
    pred_denorm_real = denormalize(predicted_real, (pkg_real['X_min'], pkg_real['X_max']))
    pred_denorm_imag = denormalize(predicted_imag, (pkg_imag['X_min'], pkg_imag['X_max']))

    # 4. Reconstruct the Complex OFDM Signals
    original_complex = test_X_real.numpy() + 1j * test_X_imag.numpy()
    clipped_complex = test_Y_real.numpy() + 1j * test_Y_imag.numpy()
    predicted_complex = pred_denorm_real + 1j * pred_denorm_imag
elif train_size < 100:
    # 2. Setup the Test Data (Files train_size-99)
    test_dataset_real = Subset(FastOFDMDataset(pt_dir, part='real'), range(train_size, 100))
    test_dataset_imag = Subset(FastOFDMDataset(pt_dir, part='imag'), range(train_size, 100))

    # We set shuffle=False to ensure real and imag batches stay perfectly aligned
    test_loader_real = DataLoader(test_dataset_real, batch_size=None, shuffle=False)
    test_loader_imag = DataLoader(test_dataset_imag, batch_size=None, shuffle=False)

    # 3. Generate Predictions for all test_size test files
    all_original, all_clipped, all_predicted = [], [], []

    test_loss_real = 0.0
    test_loss_imag = 0.0
    with torch.no_grad():
        for (X_real, Y_real, X_r_min, X_r_max, _, _), (X_imag, Y_imag, X_i_min, X_i_max, _, _) in zip(test_loader_real, test_loader_imag):
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
            pred_denorm_real = denormalize(pred_real, (X_r_min, X_r_max))
            pred_denorm_imag = denormalize(pred_imag, (X_i_min, X_i_max))

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

    print(f"\n--- Test Set Metrics ({test_size} Unseen Files) ---")
    print(f"Average Real Module MSE Loss: {avg_test_loss_real:.6f}")
    print(f"Average Imag Module MSE Loss: {avg_test_loss_imag:.6f}")

    # 4. Concatenate the test_size batches into one massive array for evaluation
    original_complex = np.concatenate(all_original, axis=0)
    clipped_complex = np.concatenate(all_clipped, axis=0)
    predicted_complex = np.concatenate(all_predicted, axis=0)

    print(f"Testing complete! Aggregated {test_size * samples_per_L} OFDM symbols.")

# -----------------------------------------------------------------------------

# 5. Calculate PAPR and CM for CCDF
orig_papr, clip_papr, pred_papr = [], [], []
orig_cm, clip_cm, pred_cm = [], [], []

# Iterate through the arrays
for i in range(test_size * samples_per_L if not one_batch else samples_per_L):
    orig_papr.append(calculate_papr(original_complex[i]))
    clip_papr.append(calculate_papr(clipped_complex[i]))
    pred_papr.append(calculate_papr(predicted_complex[i]))

    orig_cm.append(calculate_cm(original_complex[i]))
    clip_cm.append(calculate_cm(clipped_complex[i]))
    pred_cm.append(calculate_cm(predicted_complex[i]))

orig_papr, clip_papr, pred_papr = np.array(orig_papr), np.array(clip_papr), np.array(pred_papr)
orig_cm, clip_cm, pred_cm = np.array(orig_cm), np.array(clip_cm), np.array(pred_cm)

# 6. Plot the CCDF
title = f"NNICF Predicted OFDM\n{params}"
labels = ['Original OFDM', 'Clipped OFDM', 'NNICF Predicted OFDM']

plot_ccdf_compare([orig_papr, clip_papr, pred_papr], f"Original vs Clipped vs {title}", labels)
plot_ccdf_compare([orig_cm, clip_cm, pred_cm], f"Original vs Clipped vs {title}", labels, metric="CM")

# todo: find a way to embed the floor part into the plotting function
# 1. Define the target y-levels (probabilities)
papr_target_y = 1e-4
cm_target_y = 1e-3

# 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
papr_percentile = (1.0 - papr_target_y) * 100.0
cm_percentile = (1.0 - cm_target_y) * 100.0

# 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
# (This completely replaces the need for y_axis, np.where, and manual sorting!)
papr_vlines = [
    np.percentile(orig_papr, papr_percentile),
    np.percentile(clip_papr, papr_percentile)
]

cm_vlines = [
    np.percentile(orig_cm, cm_percentile),
    np.percentile(clip_cm, cm_percentile)
]

# 4. Plot!
# papr_image_path = os.path.join(graph_dir, f"{mod}_papr_{train_test_str}_{lr_str}.png")
plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

# cm_image_path = os.path.join(graph_dir, f"{mod}_cm_{train_test_str}_{lr_str}.png")
plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)

end = time.time()
# ~98s at test_size = 20
# ~6s at one_batch != None
print(f"\nTotal execution time: {end - start:.2f} seconds")
