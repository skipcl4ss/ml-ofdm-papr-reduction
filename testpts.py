import numpy as np
from ofdm.metrics import calculate_papr, calculate_cm
from ofdm.plots import plot_ccdf, plot_ccdf_compare
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from nnscf import NNSCFMapper, device, denormalize, criterion
import os
import time

# ! this file proves that the problem is most probably in the normalization in gemini branch, as when neglecting nn results, raw scf is plotted correctly while normalized and denormalized scf data isnt

one_batch = "99"
batch_suffix = f"Batch #{one_batch}"
mod = "16qam"
tech = "scf"
opt = "LBFGS"
samples_per_L = 10000
data_size = 100
train_size = 70
val_size = 10
test_size = data_size - train_size - val_size
train_val_test = f"{train_size}_{val_size}_{test_size}" if val_size else f"{train_size}_{test_size}"
lr = 0.05
lr_list = str(float(lr)).split(".")
lr_str = f"dot{lr_list[1]}" if lr < 1 else f"{lr_list[0]}dot{lr_list[1]}"
# params = f"{opt} optimizer {tech.upper()} {mod.upper()} lr = {lr}\n({train_size} Training/{val_size} Validation/{test_size} Testing){" (" + batch_suffix + ")" if batch_suffix else ''}"
# raw_dir = "./raw dir/"
# norm_dir = "./norm dir/"
# relev_dir = "./relevant files/"
norm1 = "./test dir/"
norm2 = "./test dir 2/"
relev_dir = "./test dir/"

# checkpoint_re = torch.load(
#     os.path.join(relev_dir, f"{opt}_{mod}_{tech}_{train_val_test}_{lr_str}_weights_limits_re.pth"),
#     weights_only=False
# )
# checkpoint_im = torch.load(
#     os.path.join(relev_dir, f"{opt}_{mod}_{tech}_{train_val_test}_{lr_str}_weights_limits_im.pth"),
#     weights_only=False
# )

# 1. Load the Saved Models
Test_Mod_Re = NNSCFMapper().to(device)
Test_Mod_Im = NNSCFMapper().to(device)

Test_Mod_Re.load_state_dict(torch.load(os.path.join(relev_dir, f"{opt}_{mod}_{tech}_mod_re_weights_{train_val_test}_{lr_str}.pth"), weights_only=True))
Test_Mod_Im.load_state_dict(torch.load(os.path.join(relev_dir, f"{opt}_{mod}_{tech}_mod_im_weights_{train_val_test}_{lr_str}.pth"), weights_only=True))
# Test_Mod_Re.load_state_dict(checkpoint_re['model_state_dict'])
# Test_Mod_Im.load_state_dict(checkpoint_im['model_state_dict'])


Test_Mod_Re.eval()
Test_Mod_Im.eval()

# X_r_min = checkpoint_re['X_min']
# X_r_max = checkpoint_re['X_max']
# Y_r_min = checkpoint_re['Y_min']
# Y_r_max = checkpoint_re['Y_max']
#
# X_i_min = checkpoint_im['X_min']
# X_i_max = checkpoint_im['X_max']
# Y_i_min = checkpoint_im['Y_min']
# Y_i_max = checkpoint_im['Y_max']





for one_batch in range(test_size):
    batch_suffix = f"Batch #{one_batch}"
    params = f"{opt} optimizer {tech.upper()} {mod.upper()} lr = {lr}\n({train_size} Training/{val_size} Validation/{test_size} Testing){" (" + batch_suffix + ")" if batch_suffix else ''}"

    # 2. Grab one batch of data to test (Load the dictionary packages)
    # pkg_real = torch.load(os.path.join(norm_dir, f"{mod}_{tech}_part_{one_batch:02}_real.pt"), weights_only=False)
    # pkg_imag = torch.load(os.path.join(norm_dir, f"{mod}_{tech}_part_{one_batch:02}_imag.pt"), weights_only=False)
    # pkg_real_raw = torch.load(os.path.join(raw_dir, f"{mod}_{tech}_part_{one_batch:02}_real.pt"), weights_only=False)
    # pkg_imag_raw = torch.load(os.path.join(raw_dir, f"{mod}_{tech}_part_{one_batch:02}_imag.pt"), weights_only=False)
    pkg_real_norm1 = torch.load(os.path.join(norm1, f"{mod}_{tech}_part_{one_batch:02}_real.pt"), weights_only=False)
    pkg_imag_norm1 = torch.load(os.path.join(norm1, f"{mod}_{tech}_part_{one_batch:02}_imag.pt"), weights_only=False)
    pkg_real_norm2 = torch.load(os.path.join(norm2, f"{mod}_{tech}_part_{one_batch:02}_real.pt"), weights_only=False)
    pkg_imag_norm2 = torch.load(os.path.join(norm2, f"{mod}_{tech}_part_{one_batch:02}_imag.pt"), weights_only=False)
    # # Extract tensors
    # # test_X_real, test_Y_real = pkg_real['X_norm'], pkg_real['Y_norm']
    # # test_X_imag, test_Y_imag = pkg_imag['X_norm'], pkg_imag['Y_norm']
    # test_X_real = denormalize(pkg_real['X_norm'], (X_r_min, X_r_max))
    # test_Y_real = denormalize(pkg_real['Y_norm'], (Y_r_min, Y_r_max))
    # test_X_imag = denormalize(pkg_imag['X_norm'], (X_i_min, X_i_max))
    # test_Y_imag = denormalize(pkg_imag['Y_norm'], (Y_i_min, Y_i_max))
    # test_X_real_raw = pkg_real_raw['X_raw']
    # test_Y_real_raw = pkg_real_raw['Y_raw']
    # test_X_imag_raw = pkg_imag_raw['X_raw']
    # test_Y_imag_raw = pkg_imag_raw['Y_raw']
    test_X_real_norm1 = pkg_real_norm1['X_norm']
    test_Y_real_norm1 = pkg_real_norm1['Y_norm']
    test_X_imag_norm1 = pkg_imag_norm1['X_norm']
    test_Y_imag_norm1 = pkg_imag_norm1['Y_norm']
    test_X_real_norm2 = pkg_real_norm2['X_norm']
    test_Y_real_norm2 = pkg_real_norm2['Y_norm']
    test_X_imag_norm2 = pkg_imag_norm2['X_norm']
    test_Y_imag_norm2 = pkg_imag_norm2['Y_norm']


    # 3. Generate Predictions (Reshape, Predict, Reshape back)
    with torch.no_grad():
        # Memoryless flattening
        # X_real_flat = test_X_real.view(-1, 1).to(device)
        # X_imag_flat = test_X_imag.view(-1, 1).to(device)
        X_real_flat_norm1 = test_X_real_norm1.view(-1, 1).to(device)
        X_imag_flat_norm1 = test_X_imag_norm1.view(-1, 1).to(device)
        X_real_flat_norm2 = test_X_real_norm2.view(-1, 1).to(device)
        X_imag_flat_norm2 = test_X_imag_norm2.view(-1, 1).to(device)

        # predicted_real = Test_Mod_Re(X_real_flat).view_as(test_X_real).cpu().numpy()
        # predicted_imag = Test_Mod_Im(X_imag_flat).view_as(test_X_imag).cpu().numpy()
        predicted_real_norm1 = Test_Mod_Re(X_real_flat_norm1).view_as(test_X_real_norm1).cpu().numpy()
        predicted_imag_norm1 = Test_Mod_Im(X_imag_flat_norm1).view_as(test_X_imag_norm1).cpu().numpy()
        predicted_real_norm2 = Test_Mod_Re(X_real_flat_norm2).view_as(test_X_real_norm2).cpu().numpy()
        predicted_imag_norm2 = Test_Mod_Im(X_imag_flat_norm2).view_as(test_X_imag_norm2).cpu().numpy()
    #
    # # Denormalize using exact dictionary limits
    # # ? which limits should i use when using one_batch, local or global?
    # # pred_denorm_real = denormalize(predicted_real, (pkg_real['Y_min'], pkg_real['Y_max']))
    # # pred_denorm_imag = denormalize(predicted_imag, (pkg_imag['Y_min'], pkg_imag['Y_max']))
    # pred_denorm_real = denormalize(predicted_real, (Y_r_min, Y_r_max))
    # pred_denorm_imag = denormalize(predicted_imag, (Y_i_min, Y_i_max))

    # 4. Reconstruct the Complex OFDM Signals
    # original_complex = test_X_real.numpy() + 1j * test_X_imag.numpy()
    # original_complex_raw = test_X_real_raw.numpy() + 1j * test_X_imag_raw.numpy()
    # clipped_complex = test_Y_real.numpy() + 1j * test_Y_imag.numpy()
    # clipped_complex_raw = test_Y_real_raw.numpy() + 1j * test_Y_imag_raw.numpy()
    # # predicted_complex = pred_denorm_real + 1j * pred_denorm_imag
    original_complex_norm1 = test_X_real_norm1.numpy() + 1j * test_X_imag_norm1.numpy()
    original_complex_norm2 = test_X_real_norm2.numpy() + 1j * test_X_imag_norm2.numpy()
    clipped_complex_norm1 = test_Y_real_norm1.numpy() + 1j * test_Y_imag_norm1.numpy()
    clipped_complex_norm2 = test_Y_real_norm2.numpy() + 1j * test_Y_imag_norm2.numpy()
    #
    # # 5. Calculate PAPR and CM for CCDF
    # # orig_papr, clip_papr, pred_papr = [], [], []
    # # orig_cm, clip_cm, pred_cm = [], [], []
    # orig_cm, orig_cm_raw, clip_cm, clip_cm_raw = [], [], [], []
    orig_cm_norm1, orig_cm_norm2, clip_cm_norm1, clip_cm_norm2 = [], [], [], []

    # Iterate through the arrays
    for i in range(samples_per_L):
        # orig_papr.append(calculate_papr(original_complex[i]))
        # clip_papr.append(calculate_papr(clipped_complex[i]))
        # pred_papr.append(calculate_papr(predicted_complex[i]))

        # orig_cm.append(calculate_cm(original_complex[i]))
        # orig_cm_raw.append(calculate_cm(original_complex_raw[i]))
        # clip_cm.append(calculate_cm(clipped_complex[i]))
        # clip_cm_raw.append(calculate_cm(clipped_complex_raw[i]))
        # # pred_cm.append(calculate_cm(predicted_complex[i]))
        orig_cm_norm1.append(calculate_cm(original_complex_norm1[i]))
        orig_cm_norm2.append(calculate_cm(original_complex_norm2[i]))
        clip_cm_norm1.append(calculate_cm(clipped_complex_norm1[i]))
        clip_cm_norm2.append(calculate_cm(clipped_complex_norm2[i]))

    # # orig_papr, clip_papr, pred_papr = np.array(orig_papr), np.array(clip_papr), np.array(pred_papr)
    # # orig_cm, clip_cm, pred_cm = np.array(orig_cm), np.array(clip_cm), np.array(pred_cm)
    # orig_cm, orig_cm_raw, clip_cm, clip_cm_raw = np.array(orig_cm), np.array(orig_cm_raw), np.array(clip_cm), np.array(clip_cm_raw)
    orig_cm_norm1, orig_cm_norm2, clip_cm_norm1, clip_cm_norm2 = np.array(orig_cm_norm1), np.array(orig_cm_norm2), np.array(clip_cm_norm1), np.array(clip_cm_norm2)

    # 6. Plot the CCDF
    title = f'NN{tech.upper()} Predicted OFDM\n{params}'
    # # labels = ['Original OFDM', f'{tech.upper()}', f'NN{tech.upper()} Predicted OFDM']
    # labels = ['Original OFDM', 'OG RAW', f'{tech.upper()}', f'RAW']
    labels = ['og incorrect norm', f'og correct norm', f'cf incorrect norm', f'cf correct norm']
    # # papr_list = [orig_papr, clip_papr, pred_papr]
    # # cm_list = [orig_cm, clip_cm, pred_cm]
    # cm_list = [orig_cm, orig_cm_raw, clip_cm, clip_cm_raw]
    cm_list = [orig_cm_norm1, orig_cm_norm2, clip_cm_norm1, clip_cm_norm2]

    # fixme: both percentile and max are not the most efficient solutions

    # todo: find a way to embed the floor part into the plotting function
    # 1. Define the target y-levels (probabilities)
    # papr_target_y = 1e-4
    cm_target_y = 1e-3

    # 2. Convert CCDF y-level to a standard percentile (e.g., 1e-4 becomes 99.99)
    # papr_percentile = (1.0 - papr_target_y) * 100.0
    cm_percentile = (1.0 - cm_target_y) * 100.0

    # # 3. Extract the exact x-axis values (PAPR/CM) where the line cuts the graph
    # # (This completely replaces the need for y_axis, np.where, and manual sorting!)
    # papr_vlines = [
    #     np.percentile(orig_papr, papr_percentile),
    #     np.percentile(clip_papr, papr_percentile),
    #     np.percentile(pred_papr, papr_percentile)
    # ]

    # cm_vlines = [
    #     np.percentile(orig_cm, cm_percentile),
    #     np.percentile(clip_cm, cm_percentile),
    #     # np.percentile(pred_cm, cm_percentile)
    # ]

    # cm_vlines = [
    #     np.max(orig_cm),
    #     np.max(clip_cm),
    #     np.max(pred_cm)
    # ]

    # # 4. Plot!
    # # papr_image_path = os.path.join(relev_dir, f"{opt}_{mod}_{train_val_test}_{lr_str}_papr")
    # plot_ccdf_compare(papr_list, f'Original vs {tech.upper()} vs {title}', labels)
    # plot_ccdf(pred_papr, title, metric="papr", vlines=papr_vlines)

    cm_image_path = os.path.join(relev_dir, f"{opt}_{mod}_{tech}_{train_val_test}_{lr_str}_cm")
    plot_ccdf_compare(cm_list, f'Original vs {tech.upper()} vs {title}', labels, metric="CM")
    # plot_ccdf(pred_cm, title, metric="cm", vlines=cm_vlines)