import numpy as np
import matplotlib.pyplot as plt
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

    # fixme: both percentile and max are not the most effecient solution
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

    if label:
        plt.legend(loc='lower left')

    ccdf_format(title, metric, vlines)
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
    plt.legend(loc='lower left', ncol=2 if len(val_list) > 3 else 1)
    ccdf_format(title, metric)
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
        elif any(k in name for k in ("original", "unclipped")):
            plt.semilogy(EbNo_range, vals, color="k", linewidth=2, marker="s", label=label[i])
        elif any(k in name for k in ("filtered", "icf")):
            plt.semilogy(EbNo_range, vals, color="r", linewidth=2, marker="o", label=label[i])
        # elif "clipped" in label[i].lower():
        #     plt.semilogy(EbNo_range, vals, color="b", linewidth=2, marker="*", label=label[i])
        else:
            plt.semilogy(EbNo_range, vals, linewidth=2, label=label[i])

    # # theoretical part
    # y_theory = np.array([ber_theoretical(EbNo_dB, M) for EbNo_dB in EbNo_range])
    # plt.semilogy(EbNo_range, y_theory, '--', color='black', linewidth=2, label='Theoretical')

    plt.legend(loc='lower left', ncol=2 if len(val_list) > 3 else 1)
    ber_format(title, M)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
