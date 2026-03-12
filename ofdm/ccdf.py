import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def theoretical_CCDF(N, papr_db_range):
    # CDF = (1 - e^(-z^2)) ^ N
    # CCDF = 1 - CDF
    gamma = 10 ** (papr_db_range / 10) # ? is this z^2
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

def plot_CCDF(papr_dict, N, title_suffix=''):
    plt.figure(figsize=(10, 7))
    sorted_papr = np.sort(papr_dict)
    y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr) # ? why dont we use the theoretical CCDF function
    print(sorted_papr)
    print(len(sorted_papr))
    print(y_axis)
    # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
    plt.semilogy(sorted_papr, y_axis, linewidth=2, label='Unclipped')

    # all_data = np.concatenate([v for v in papr_dict if v.size > 0]) if papr_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_CCDF(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plot_formatting(title_suffix)
    plt.show()

def plot_CCDF_compare(papr_dict, N, title_suffix=''):
    plt.figure(figsize=(10, 7))
    colors = cm.viridis(np.linspace(0, 0.9, len(papr_dict)))
    for i, idx in enumerate(papr_dict):
        sorted_papr = np.sort(idx)
        y_axis = np.arange(len(sorted_papr), 0, -1) / len(sorted_papr)  # ? why dont we use the theoretical CCDF function
        # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
        if i == 0:
            # plt.semilogy(sorted_papr, y_axis, color=colors[0], linewidth=2, label='Unclipped')
            plt.semilogy(sorted_papr, y_axis, linewidth=2, label='Unclipped')
        else:
            # plt.semilogy(sorted_papr, y_axis, color=colors[i], linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')
            plt.semilogy(sorted_papr, y_axis, linewidth=2, label=f'ICF ({i} iteration{"s" if i > 1 else ""})')

    # all_data = np.concatenate([v for v in papr_dict.values() if v.size > 0]) if papr_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = theoretical_CCDF(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    plot_formatting(title_suffix)
    plt.show()
