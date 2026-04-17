import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from metrics import ccdf_theoretical

def ccdf_format(title, metric='PAPR', vlines=None):
    # Define boundaries
    if metric.lower() == 'papr':
        bottom = 1e-4
        plt.xlabel('PAPR$_0$ [dB]')
        plt.ylabel('Pr(PAPR > PAPR$_0$)')
        plt.axhline(y=1e-3, color='gray', alpha=0.4)
    elif metric.lower() == 'cm':
        bottom = 1e-3
        plt.xlabel('Cubic Metric [dB]')
        plt.ylabel('CCDF')

    if vlines:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', alpha=0.4)

    # plt.xlim([left, right])
    plt.ylim([bottom, 1])

    if title:
        plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(y=1e-1, color='gray', alpha=0.4)
    plt.axhline(y=1e-2, color='gray', alpha=0.4)
    # plt.legend(loc='lower left')
    # plt.tight_layout()

def ber_format(EbNo_range, title):
    plt.xlabel('E$_b/N$_0 [dB]')
    plt.ylabel('BER')

    if title:
        plt.title(title)
    plt.xlim([EbNo_range[0], EbNo_range[-1]])

    # todo: implement ylim() in ber
    plt.ylim([1e-4, 1e0])
    # if mod == "16qam":
    #     plt.ylim([1e-3, 1e0])
    # elif mod == "qpsk":
    #     plt.ylim([1e-4, 1e0])
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # plt.tight_layout()

def plot_ccdf(vals, title=None, label=None, metric='PAPR', vlines=None, save=False):
    # Define boundaries
    if metric.lower() == "papr":
        left, right = 4, 12
        bottom = 1e-4
    elif metric.lower() == "cm":
        left, right = 2.5, 6
        bottom = 1e-3

    plt.figure(figsize=(10, 7))
    sorted_vals = np.sort(vals)
    # ? why dont we use the theoretical CCDF function
    y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals)

    # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
    plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label)

    # all_data = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = ccdf_theoretical(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    # ? is percentile() 100% accurate
    p = (1.0 - bottom) * 100.0
    percentile = np.percentile(vals, p, method="higher")
    if left < percentile <= right:
        plt.xlim([left, right])
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

        # todo: Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default
        # plt.semilogy(sorted_vals, y_axis, color=colors[i], linewidth=2, label=label[i])
        plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label[i])

    # all_data = np.concatenate([v for v in val_list.values() if v.size > 0]) if val_list else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = ccdf_theoretical(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plt.xlim([left, right])
    plt.legend(loc='lower left')
    ccdf_format(title, metric)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

# todo: complete the colors part
def plot_ber(EbNo_range, val_list, title, label=None, save=False):
    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(val_list)))
    # colors = cm.plasma(np.linspace(0, 0.9, len(val_list)))
    if not label:
        l = [f'Clipped Iteration {i}' for i in range(len(val_list))]
        l += [f'Clipped + Filtered Iteration {i}' for i in range(len(val_list))]
        label = ["Unclipped", "Theoretical"] + l

    for i, vals in enumerate(val_list):
        # todo: add theoretical part
        # plt.semilogy(EbNo_range, val_list[i], color=colors[i], linewidth=2, label=label[i])
        plt.semilogy(EbNo_range, vals, linewidth=2, label=label[i])

    plt.legend(loc='lower left')
    ber_format(EbNo_range, title)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
