import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from metrics import ccdf_theoretical

def ccdf_format(title, metric='PAPR', vlines=None):
    if metric.lower() == 'papr':
        # plt.xlim([4, 12])
        plt.xlabel('PAPR$_0$ [dB]')
        plt.ylim([1e-4, 1])
        plt.ylabel('Pr(PAPR > PAPR$_0$)')
        plt.axhline(y=1e-3, color='gray', alpha=0.4)
    elif metric.lower() == 'cm':
        # plt.xlim([2.5, 6])
        plt.xlabel('Cubic Metric [dB]')
        plt.ylim([1e-3, 1])
        plt.ylabel('CCDF')

    if vlines:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', alpha=0.4)

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

    # todo: implement this
    # if mod == "16qam":
    #     plt.ylim([1e-3, 1e0])
    # elif mod == "qpsk":
    #     plt.ylim([1e-4, 1e0])
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.ylim([1e-4, 1e0])
    # plt.tight_layout()

def plot_ccdf(vals, title, label, metric='PAPR', vlines=None, save=False):
    plt.figure(figsize=(10, 7))
    sorted_vals = np.sort(vals)
    y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals) # ? why dont we use the theoretical CCDF function
    # y_axis = 1 + 1 / len(sorted_vals) - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) # ! this line is essentially the same as the one above, but in the format of the one below
    # y_axis = 1 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) # ? why not this

    # Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default solid
    plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label)

    # all_data = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = ccdf_theoretical(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    if metric.lower() == "papr" and 4 <= max(sorted_vals) <= 12:
        plt.xlim([4, 12])
    elif metric.lower() == "papr":
        plt.axvline(x=4, color='gray', alpha=0.4)
        plt.axvline(x=12, color='gray', alpha=0.4)
    elif metric.lower() == "cm" and 2.5 <= max(sorted_vals) <= 6:
        plt.xlim([2.5, 6])
    elif metric.lower() == "cm":
        plt.axvline(x=2.5, color='gray', alpha=0.4)
        plt.axvline(x=6, color='gray', alpha=0.4)

    if label:
        plt.legend(loc='lower left')

    ccdf_format(title, metric, vlines)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

def plot_ccdf_compare(val_dict, title, label=None, metric='PAPR', save=False):
    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(val_dict)))
    if not label:
        label = [f'ICF ({i} iteration{"s" if i > 1 else ""})' for i in range(len(val_dict))]
        label = ["Unclipped"] + label

    for i, vals in enumerate(val_dict):
        sorted_vals = np.sort(vals)
        y_axis = np.arange(len(sorted_vals), 0, -1) / len(sorted_vals)

        # todo: Clipped (no filtering) -> keep solid; Clipped + Filtered -> dashed; others default
        # plt.semilogy(sorted_vals, y_axis, color=colors[i], linewidth=2, label=label[i])
        plt.semilogy(sorted_vals, y_axis, linewidth=2, label=label[i])

    # all_data = np.concatenate([v for v in val_dict.values() if v.size > 0]) if val_dict else np.array([])
    # if all_data.size > 0:
    #     x_theory = np.linspace(np.min(all_data), np.max(all_data) + 2, 200)
    #     y_theory = ccdf_theoretical(N, x_theory)
    #     plt.semilogy(x_theory, y_theory, 'k--', linewidth=1.5, label='Theoretical (L=1)')

    if metric.lower() == 'papr':
        plt.xlim([4, 12])
    elif metric.lower() == 'cm':
        plt.xlim([2.5, 6])

    plt.legend(loc='lower left')
    ccdf_format(title, metric)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()

# todo: complete the colors part
def plot_ber(EbNo_range, val_dict, title, label=None, save=False):
    plt.figure(figsize=(10, 7))
    # colors = cm.viridis(np.linspace(0, 0.9, len(val_dict)))
    # colors = cm.plasma(np.linspace(0, 0.9, len(val_dict)))
    if not label:
        l = [f'Clipped Iteration {i}' for i in range(len(val_dict))]
        l += [f'Clipped + Filtered Iteration {i}' for i in range(len(val_dict))]
        label = ["Unclipped", "Theoretical"] + l

    for i, vals in enumerate(val_dict):
        # todo: add theoretical part
        # plt.semilogy(EbNo_range, val_dict[i], color=colors[i], linewidth=2, label=label[i])
        plt.semilogy(EbNo_range, vals, linewidth=2, label=label[i])

    plt.legend(loc='lower left')
    ber_format(EbNo_range, title)
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
