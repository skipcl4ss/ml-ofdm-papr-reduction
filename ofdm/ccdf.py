import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm

def theoretical_ccdf(N, papr_dB_range):
    # CDF = (1 - e^(-z^2)) ^ N
    # CCDF = 1 - CDF
    gamma = 10 ** (papr_dB_range / 10) # ? is this z^2
    ccdf = 1 - (1 - np.exp(-gamma)) ** N
    return ccdf

def plot_formatting(title_suffix='', metric='PAPR', vlines=None):
    if metric.lower() == 'papr':
        # plt.xlim([4, 12])
        plt.xlabel('PAPR$_0$ [dB]')
        plt.ylim([1e-4, 1])
        plt.ylabel('Pr(PAPR > PAPR$_0$)')
        plt.axhline(y=1e-3, color='gray', linestyle=':')
    elif metric.lower() == 'cm':
        # plt.xlim([2.5, 6])
        plt.xlabel('Cubic Metric [dB]')
        plt.ylim([1e-3, 1])
        plt.ylabel('CCDF')

    if vlines:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', linestyle=':')

    plt.title(title_suffix)
    plt.grid(True, which='both')
    plt.axhline(y=1e-1, color='gray', linestyle=':')
    plt.axhline(y=1e-2, color='gray', linestyle=':')
    # plt.axhline(y=1e-4, color='gray', linestyle=':')
    plt.legend(loc='lower left')
    # plt.tight_layout()

def plot_ccdf(val_dict, title_suffix='', label=None, metric='PAPR', vlines=None, save=False):
    plt.figure(figsize=(10, 7))
    sorted_vals = np.sort(val_dict)
    y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals) # ? why dont we use the theoretical CCDF function
    # y_axis = 1 + 1 / len(sorted_vals) - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) # ! this line is essentially the same as the one above, but in the format of the one below
    # y_axis = 1 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) # ? why not this

    # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
    plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label if label else title_suffix)

    # all_data = np.concatenate([v for v in val_dict if v.size > 0]) if val_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_ccdf(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    if metric.lower() == "papr" and 4 <= max(sorted_vals) <= 12:
        plt.xlim([4, 12])
    elif metric.lower() == "papr":
        plt.axvline(x=4, color='gray', linestyle=':')
        plt.axvline(x=12, color='gray', linestyle=':')
    elif metric.lower() == "cm" and 2.5 <= max(sorted_vals) <= 6:
        plt.xlim([2.5, 6])
    elif metric.lower() == "cm":
        plt.axvline(x=2.5, color='gray', linestyle=':')
        plt.axvline(x=6, color='gray', linestyle=':')
    plot_formatting(title_suffix, metric, vlines)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

def plot_ccdf_compare(val_dict, title_suffix='', label=None, metric='PAPR', vlines=None, save=False):
    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(val_dict)))
    for i, idx in enumerate(val_dict):
        sorted_vals = np.sort(idx)
        y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals)  # ? why dont we use the theoretical CCDF function

        # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default
        # todo: edit the function so that it takes labels as input
        plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label[i] if label else '')
        # if i == 0:
        #     # plt.semilogy(sorted_vals, y_axis, color=colors[0], linewidth=2, label='Unclipped')
        #     plt.semilogy(sorted_vals, y_axis, linewidth=2, label='Unclipped')
        # else:
        #     # plt.semilogy(sorted_vals, y_axis, color=colors[i], linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')
        #     plt.semilogy(sorted_vals, y_axis, linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')

    # all_data = np.concatenate([v for v in val_dict.values() if v.size > 0]) if val_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_ccdf(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    if metric.lower() == 'papr':
        plt.xlim([4, 12])
    elif metric.lower() == 'cm':
        plt.xlim([2.5, 6])
    plot_formatting(title_suffix, metric, vlines)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
