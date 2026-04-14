import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
from ofdm.plots import plot_ccdf, plot_ccdf_compare
from ofdm.metrics import calculate_papr, calculate_cm
from nnicf import NNICFMapper, device, denormalize

# -----------------------------------------------------------------------------

start = time.time()
samples_per_L = 10000

# 1024 features based on N=256 and oversampling L=4
input_features = 256 * 4

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
train_size = 100
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
#%%
# Explicitly tell PyTorch to utilize your 8 CPU cores for matrix math
# Check for GPU availability to drastically speed up training
# ! cuda is available only on nvidia gpu
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        X_norm, Y_norm = torch.load(file_path, weights_only=True)
        return X_norm, Y_norm

# -----------------------------------------------------------------------------

# Standard Mean Squared Error loss
criterion = nn.MSELoss()

# -----------------------------------------------------------------------------

# 1. Load the Saved Models
Test_Mod_Re = NNICFMapper(input_features).to(device)
Test_Mod_Im = NNICFMapper(input_features).to(device)

Test_Mod_Re.load_state_dict(torch.load(f"./trained_models/{mod}_mod_re_weights_{train_test_str}_{lr_str}.pth", weights_only=True))
Test_Mod_Im.load_state_dict(torch.load(f"./trained_models/{mod}_mod_im_weights_{train_test_str}_{lr_str}.pth", weights_only=True))

Test_Mod_Re.eval()
Test_Mod_Im.eval()

# -----------------------------------------------------------------------------

if one_batch:
    # 2. Grab one batch of data to test
    test_X_real, test_Y_real = torch.load(f"./data_pt/{mod}_tx_rx_32_part_{one_batch}_real.pt", weights_only=True)
    test_X_imag, test_Y_imag = torch.load(f"./data_pt/{mod}_tx_rx_32_part_{one_batch}_imag.pt", weights_only=True)

    # 3. Generate Predictions (No gradients needed for testing)
    with torch.no_grad():
        predicted_real = Test_Mod_Re(test_X_real).numpy()
        predicted_imag = Test_Mod_Im(test_X_imag).numpy()

    # fixme: get minmax values
    pred_denorm_real = denormalize(predicted_real)
    pred_denorm_imag = denormalize(predicted_imag)

    # done: denormalize before recombining
    # 4. Reconstruct the Complex OFDM Signals
    original_complex = test_X_real.numpy() + 1j * test_X_imag.numpy()
    clipped_complex = test_Y_real.numpy() + 1j * test_Y_imag.numpy()
    # predicted_complex = predicted_real + 1j * predicted_imag
    predicted_complex = pred_denorm_real + 1j * pred_denorm_imag
elif train_size < 100:
    # 2. Setup the Test Data (Files train_size-99)
    test_dataset_real = Subset(FastOFDMDataset("./data_pt/", part='real'), range(train_size, 100))
    test_dataset_imag = Subset(FastOFDMDataset("./data_pt/", part='imag'), range(train_size, 100))

    # We set shuffle=False to ensure real and imag batches stay perfectly aligned
    test_loader_real = DataLoader(test_dataset_real, batch_size=None, shuffle=False)
    test_loader_imag = DataLoader(test_dataset_imag, batch_size=None, shuffle=False)

    # 3. Generate Predictions for all test_size test files
    all_original, all_clipped, all_predicted = [], [], []

    test_loss_real = 0.0
    test_loss_imag = 0.0
    with torch.no_grad():
        for (X_real, Y_real), (X_imag, Y_imag) in zip(test_loader_real, test_loader_imag):
            # Move inputs to device
            X_real, X_imag = X_real.to(device), X_imag.to(device)

            # Predict
            # ! shouldnt use .numpy() method because of criterion
            pred_real = Test_Mod_Re(X_real)
            pred_imag = Test_Mod_Im(X_imag)

            # Calculate the Loss for this batch
            loss_real = criterion(pred_real, Y_real)
            loss_imag = criterion(pred_imag, Y_imag)

            test_loss_real += loss_real.item()
            test_loss_imag += loss_imag.item()

            # fixme: get minmax values
            pred_denorm_real = denormalize(pred_real)
            pred_denorm_imag = denormalize(pred_imag)

            # done: denormalize before recombining
            # Reconstruct complex signals
            orig_complex = X_real.numpy() + 1j * X_imag.numpy()
            clip_complex = Y_real.numpy() + 1j * Y_imag.numpy()
            # pred_complex = pred_real + 1j * pred_imag
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
# ! the size of data had an axis of size 10k, which i assume is samples_per_L
y_axis_len = test_size * samples_per_L if not one_batch else samples_per_L
y_axis = np.arange(y_axis_len, 0, -1) / y_axis_len
papr_floor = np.where(y_axis == 1e-4)[0]
cm_floor = np.where(y_axis == 1e-3)[0]

papr_vlines = [np.sort(orig_papr)[papr_floor], np.sort(clip_papr)[papr_floor]]
# papr_image_path = f'./nngraphs/{mod}_papr_{train_test_str}_{lr_str}.png'
plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

cm_vlines = [np.sort(orig_cm)[cm_floor], np.sort(clip_cm)[cm_floor]]
# cm_image_path = f'./nngraphs/{mod}_cm_{train_test_str}_{lr_str}.png'
plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)

end = time.time()
# ~98s at test_size = 20
# ~6s at one_batch != None
print(f"\nTotal execution time: {end - start:.2f} seconds")
