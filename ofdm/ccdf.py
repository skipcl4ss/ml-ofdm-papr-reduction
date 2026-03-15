import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm

def theoretical_ccdf(N, papr_dB_range):
    # CDF = (1 - e^(-z^2)) ^ N
    # CCDF = 1 - CDF
    gamma = 10 ** (papr_dB_range / 10) # ? is this z^2
    ccdf = 1 - (1 - np.exp(-gamma)) ** N
    return ccdf

def plot_formatting(title_suffix=''):
    plt.xlim([4, 12])
    plt.ylim([1e-5, 1])
    plt.xlabel('PAPR$_0$ [dB]')
    plt.ylabel('Pr(PAPR > PAPR$_0$)')
    plt.title(title_suffix)
    plt.grid(True, which='both')
    plt.axhline(y=1e-1, color='black', linestyle=':')
    plt.axhline(y=1e-2, color='black', linestyle=':')
    plt.axhline(y=1e-3, color='black', linestyle=':')
    plt.axhline(y=1e-4, color='black', linestyle=':')
    plt.legend(loc='lower left')
    # plt.tight_layout()

def plot_ccdf(papr_dict, title_suffix='', label=None):
    plt.figure(figsize=(10, 7))
    sorted_papr = np.sort(papr_dict)
    y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr) # ? why dont we use the theoretical CCDF function
    # y_axis = 1 + 1 / len(sorted_papr) - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr) # ! this line is essentially the same as the one above, but in the format of the one below
    # y_axis = 1 - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr) # ? why not this

    # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
    plt.semilogy(sorted_papr, y_axis, linewidth=2, label=label if label else title_suffix)

    # all_data = np.concatenate([v for v in papr_dict if v.size > 0]) if papr_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_ccdf(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plot_formatting(title_suffix)
    plt.show()

def plot_ccdf_compare(papr_dict, title_suffix='', label=None):
    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(papr_dict)))
    for i, idx in enumerate(papr_dict):
        sorted_papr = np.sort(idx)
        y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr)  # ? why dont we use the theoretical CCDF function

        # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default
        # todo: edit the function so that it takes labels as input
        plt.semilogy(sorted_papr, y_axis, linewidth=2, label=label[i] if label else '')
        # if i == 0:
        #     # plt.semilogy(sorted_papr, y_axis, color=colors[0], linewidth=2, label='Unclipped')
        #     plt.semilogy(sorted_papr, y_axis, linewidth=2, label='Unclipped')
        # else:
        #     # plt.semilogy(sorted_papr, y_axis, color=colors[i], linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')
        #     plt.semilogy(sorted_papr, y_axis, linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')

    # all_data = np.concatenate([v for v in papr_dict.values() if v.size > 0]) if papr_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_ccdf(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plot_formatting(title_suffix)
    plt.show()
