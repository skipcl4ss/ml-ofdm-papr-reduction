import numpy as np
import matplotlib.pyplot as plt
import torch
# import matplotlib.cm as cm
# from ofdm.metrics import ccdf_theoretical
# from ofdm.metrics import ber_theoretical

def ccdf_format(title, metric='PAPR', vlines=None):
    # Define boundaries
    bottom = 1e-3
    if metric.lower() == 'papr':
        plt.xlabel('PAPR$_0$ [dB]')
        plt.ylabel('Pr(PAPR > PAPR$_0$)')
    elif metric.lower() == 'cm':
        plt.xlabel('Cubic Metric [dB]')
        plt.ylabel('CCDF')

    if vlines:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', alpha=0.4)

    plt.ylim(bottom, 1e0)

    if title:
        plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(y=1e-1, color='gray', alpha=0.4)
    plt.axhline(y=1e-2, color='gray', alpha=0.4)
    # plt.legend(loc='lower left')
    # plt.tight_layout()

def plot_ccdf(vals, title=None, label=None, metric='PAPR', vlines=None, save=False):
    # Define boundaries
    bottom = 1e-3
    if metric.lower() == "papr":
        left, right = 4, 12
    elif metric.lower() == "cm":
        left, right = 2.5, 6

    plt.figure(figsize=(10, 7))
    sorted_vals = np.sort(vals)
    # ? why dont we use the theoretical CCDF function
    y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals)

    plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label)

    # # theoretical part
    # if metric.lower() == "papr":
        # all_data = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([])
        # if all_data.size > 0:
        #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
        #     y_theory = ccdf_theoretical(N, x_theory)
        #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    # fixme: both percentile and max are not the most efficient solutions
    # ? is percentile() 100% accurate
    p = (1.0 - bottom) * 100.0
    percentile = np.percentile(vals, p, method="higher")
    if left < percentile <= right:
    # value_at_bottom = np.max(sorted_vals)
    # if left < value_at_bottom <= right:
        plt.xlim(left, right)
    else:
        plt.axvline(x=left, color='gray', alpha=0.4)
        plt.axvline(x=right, color='gray', alpha=0.4)

    ccdf_format(title, metric, vlines)

    if label:
        plt.legend(loc='lower left')
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

def plot_ccdf_compare(val_list, title=None, label=None, metric='PAPR', save=False):
    # Define boundaries
    if metric.lower() == "papr":
        left, right = 4, 12
    elif metric.lower() == "cm":
        left, right = 2.5, 6

    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(val_list)))
    if not label:
        label = [f'ICF ({i} iteration{"s" if i > 1 else ""})' for i in range(len(val_list))]
        label = ["Unclipped"] + label

    for i, vals in enumerate(val_list):
        sorted_vals = np.sort(vals)
        y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals)

        # # plt.semilogy(sorted_vals, y_axis, color=colors[i], linewidth=2, label=label[i])
        # plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label[i])
        name = label[i].lower()
        if any(k in name for k in ("nn", "neural", "predicted", "proposed")):
            plt.semilogy(sorted_vals, y_axis, color="g", linewidth=2,
                         # marker="D",
                         label=label[i])
        elif any(k in name for k in ("original", "unclipped")):
            plt.semilogy(sorted_vals, y_axis, color="k", linewidth=2,
                         # marker="s",
                         label=label[i])
        elif any(k in name for k in ("filtered", "icf")):
            plt.semilogy(sorted_vals, y_axis, color="r", linewidth=2,
                         # marker="o",
                         label=label[i])
        elif any(k in name for k in ("simplified", "scf")):
            plt.semilogy(sorted_vals, y_axis, color="b", linewidth=2,
                         # marker="^",
                         label=label[i])
        # elif "clipped" in label[i].lower():
        #     plt.semilogy(sorted_vals, y_axis, color="b", linewidth=2,
        #                  # marker="*",
        #                  label=label[i])
        else:
            plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label[i])

    # # theoretical part
    # if metric.lower() == "papr":
    #     all_data = np.concatenate([v for v in val_list if len(v) > 0]) if val_list else np.array([])
    #     if all_data.size > 0:
    #         x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #         y_theory = ccdf_theoretical(N, x_theory)
    #         plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plt.xlim(left, right)
    ccdf_format(title, metric)
    plt.legend(loc='lower left', ncol=2 if len(val_list) > 3 else 1)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

def ber_format(title, M):
    # Define boundaries
    if M == 4:
        top, bottom = 1e-1, 1e-4
        right = 9
    elif M == 16:
        top, bottom = 1e-0, 1e-3
        right = 16

    plt.xlabel('E$_b$/N$_0$ [dB]')
    plt.ylabel('BER')

    if title:
        plt.title(title)
    plt.xlim(0, right)
    plt.ylim(bottom, top)

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # plt.tight_layout()

# todo: correct the legend order and try using cm
def plot_ber(EbNo_range, val_list, title=None, label=None, M=4, save=False):
    plt.figure(figsize=(10, 7))
    # filtered = cm.viridis(np.linspace(0, 0.9, iterations))
    # clipped = cm.plasma(np.linspace(0, 0.9, iterations))
    if not label:
        label = ["Unclipped"]
        label += [f'Clipped Iteration {i + 1}' for i in range(len(val_list) // 2)]
        label += [f'Clipped + Filtered Iteration {i + 1}' for i in range(len(val_list) // 2)]
    if len(label) < len(val_list):
        label.extend(["Unlabeled"] * (len(val_list) - len(label)))

    for i, vals in enumerate(val_list):
        # if "filtered" in label[i].lower():
        #     plt.semilogy(EbNo_range, vals, "o:", color=filtered[(i - 1 - corrector) // 2], linewidth=2, label=label[i])
        # elif label[i].lower().startswith("clipped"):
        #     plt.semilogy(EbNo_range, vals, "*-", color=clipped[(i - corrector) // 2], linewidth=2, label=label[i])
        name = label[i].lower()
        if any(k in name for k in ("nn", "neural", "predicted", "proposed")):
            plt.semilogy(EbNo_range, vals, color="g", linewidth=2, marker="D", label=label[i])
        elif any(k in name for k in ("no clip", "unclipped")):
            plt.semilogy(EbNo_range, vals, color="k", linewidth=2, marker="s", label=label[i])
        elif any(k in name for k in ("filtered", "icf")):
            plt.semilogy(EbNo_range, vals, color="r", linewidth=2, marker="o", label=label[i])
        elif any(k in name for k in ("simplified", "scf")):
            plt.semilogy(EbNo_range, vals, color="b", linewidth=2, marker="^", label=label[i])
        # elif "clipped" in label[i].lower():
        #     plt.semilogy(EbNo_range, vals, color="b", linewidth=2, marker="*", label=label[i])
        else:
            plt.semilogy(EbNo_range, vals, linewidth=2, label=label[i])

    # # theoretical part
    # y_theory = np.array([ber_theoretical(EbNo_dB, M) for EbNo_dB in EbNo_range])
    # plt.semilogy(EbNo_range, y_theory, '--', color='black', linewidth=2, label='Theoretical')

    ber_format(title, M)
    plt.legend(loc='lower left', ncol=2 if len(val_list) > 3 else 1)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

def plot_constellation(symbols, mod, limit, title=None, label=None, color=None, save=False):
    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, c=color, marker='o', s=20, label=label)
    plt.title(title)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)

    if mod == "16qam":
        step = 1 / np.sqrt(10)
        l = r"{\sqrt{10}}$"
    elif mod == "qpsk":
        step = 1 / np.sqrt(2)
        l = r"{\sqrt{2}}$"
    labels = [r"$-\frac{4}" + l, r"$-\frac{2}" + l, 0, r"$\frac{2}" + l, r"$\frac{4}" + l]
    ticks = np.arange(-4 * step, 5 * step, 2 * step)
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    plt.legend(loc='lower left', ncol=2)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

# todo: add a parameter for ylimit
def plot_signals(signals, labels, clip_threshold=None, save=False):
    """
    Plot example signals in time domain.
    """
    top = np.max(np.abs(signals['original']))
    right = len(next(iter(signals.values())))
    if not clip_threshold:
        clip_threshold = np.max(np.abs(signals['clipped']))
    l = len(signals)
    x_axis = np.arange(right)

    fig, axs = plt.subplots(l, 1, figsize=(10, 8), sharex=True)

    for i, (k, v) in enumerate(signals.items()):
        axs[i].stem(x_axis, np.abs(v), basefmt=" ", markerfmt=".", linefmt="C0-")
        axs[i].set_title(labels[i])
        axs[i].grid(True)
        axs[i].set_ylabel('Amplitude')
        axs[i].set_xlim(0, right)
        axs[i].set_ylim(0, top * 1.1)
        axs[i].axhline(clip_threshold, color='gray', linestyle='--', label='Clipping Threshold')
        axs[i].legend()
        if i + 1 == l:
            plt.xlabel('Time')

    # plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

    # s_orig = np.abs(signals['original'])
    # s_clip = np.abs(signals['clipped'])
    # s_filt = np.abs(signals['filtered'])
    #
    # x_axis = np.arange(len(s_orig))
    #
    # fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    #
    # axs[0].stem(x_axis, s_orig, basefmt=" ", markerfmt=".", linefmt="C0-")
    # axs[0].set_title(f'Normal OFDM Signal)')
    # axs[0].grid(True)
    # axs[0].set_ylabel('Amplitude')
    # axs[0].set_xlim(0, len(s_orig))
    # axs[0].set_ylim(0, np.max(s_orig) * 1.1)
    # axs[0].axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # axs[0].legend()
    #
    # axs[1].stem(x_axis, s_clip, basefmt=" ", markerfmt=".", linefmt="C0-")
    # axs[1].set_title(f'Clipped OFDM Signal ({iterations} iteration{"s" if iterations > 1 else ""})')
    # axs[1].grid(True)
    # axs[1].set_ylabel('Amplitude')
    # axs[1].set_xlim(0, len(s_clip))
    # axs[1].set_ylim(0, np.max(s_orig) * 1.1)
    # axs[1].axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # axs[1].legend()
    #
    # axs[2].plot(x_axis, s_filt, color='C0')
    # axs[2].fill_between(x_axis, s_filt, color='C0', alpha=0.3)
    # axs[2].set_title('Clipped and Filtered OFDM Signal')
    # axs[1].grid(True)
    # axs[2].set_xlabel('Time')
    # axs[2].set_ylabel('Amplitude')
    # axs[2].set_xlim(0, len(s_filt))
    # axs[2].set_ylim(0, np.max(s_orig) * 1.1)
    # axs[2].axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # axs[2].legend()
    #
    # # plt.tight_layout()
    # plt.show()

# todo: add a parameter for ylimit
def plot_signals2(signals, labels, title='', clip_threshold=None, save=False):
    top = np.max(np.abs(signals['original']))
    right = len(next(iter(signals.values())))
    if not clip_threshold:
        clip_threshold = np.max(np.abs(signals['clipped']))
    l = len(signals)
    x_axis = np.arange(right)

    plt.figure(figsize=(10, 8))
    plt.title(title + "Signals in Time Domain")

    for i, (k, v) in enumerate(signals.items()):
        plt.subplot(l, 1, i + 1)
        plt.plot(x_axis, np.abs(v), label=labels[i])
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.xlim(0, right)
        plt.ylim(0, top * 1.1)
        plt.axhline(clip_threshold, color='gray', linestyle='--', label='Clipping Threshold')
        plt.legend()
        if i + 1 == l:
            plt.xlabel('Time')

    # plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

    # s_orig = np.abs(signals['original'])
    # s_clip = np.abs(signals['clipped'])
    # s_filt = np.abs(signals['filtered'])
    #
    # x_axis = np.arange(len(s_orig))
    #
    # plt.subplot(3, 1, 1)
    # plt.title('Time Domain OFDM Signal (Original vs Clipped vs Filtered)')
    # plt.plot(x_axis, s_orig, 'b', label='Original OFDM')
    # plt.grid(True)
    # plt.ylabel('Amplitude')
    # plt.xlim(0, len(s_orig))
    # plt.ylim(0, np.max(s_orig) * 1.1)
    # plt.axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # plt.legend()
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(x_axis, s_clip, 'g-', linewidth=1, label=f'Clipped ({iterations} iteration{"s" if iterations > 1 else ""})')
    # plt.grid(True)
    # plt.ylabel('Amplitude')
    # plt.xlim(0, len(s_clip))
    # plt.ylim(0, np.max(s_orig) * 1.1)
    # plt.axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # plt.legend()
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(x_axis, s_filt, 'r--', linewidth=1, label=f'Filtered ({iterations} iteration{"s" if iterations > 1 else ""})')
    # plt.grid(True)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.xlim(0, len(s_filt))
    # plt.ylim(0, np.max(s_orig) * 1.1)
    # plt.axhline(np.max(s_clip), color='gray', linestyle='--', label='Clipping Threshold')
    # plt.legend()
    #
    # # plt.tight_layout()
    # plt.show()

def plot_dataset_mapping(pt_file_path, title, raw=False, save=False):
    # Load the normalized tensor data
    if raw:
        X_str = 'X_raw'
        Y_str = 'Y_raw'
    else:
        X_str = 'X_norm'
        Y_str = 'Y_norm'

    data = torch.load(pt_file_path, weights_only=True)
    x = data[X_str].cpu().numpy().flatten()
    y = data[Y_str].cpu().numpy().flatten()

    left = x.min()
    right = x.max()
    bottom = y.min()
    top = y.max()
    # print(left == -1, bottom == -1)
    # print(right == 1, top == 1)

    # print(left, right, bottom, top)

    # Plot Input vs Target
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.1, s=1)

    # Draw a perfectly linear 1:1 reference line
    # plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', label="Linear 1:1 Mapping")
    plt.plot([left, right], [bottom, top], color='red', linestyle='--', label="Linear 1:1 Mapping")

    plt.xlim(left, right)
    plt.ylim(bottom, top)

    plt.title(f"Dataset Mapping: {title}\n({X_str} vs {Y_str})")
    plt.xlabel(f"Input Amplitude ({"Raw" if raw else "Normalized"})")
    plt.ylabel(f"Target Amplitude ({"Raw" if raw else "Normalized"})")
    plt.grid(True)
    plt.legend(loc='lower right')
    if save:
        plt.savefig(save, dpi=150)
    plt.show()