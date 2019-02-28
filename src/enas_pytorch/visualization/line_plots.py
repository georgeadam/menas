import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)


def line_plot(x, y, title, x_label, y_label, save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    for i in range(len(x)):
        sns.lineplot(x[i], y[i], ax=ax)

    ax.set_ylim(0.0, 1.1)

    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)

    # fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path)


def entropy_plot(x, y, std, save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(x, y)
    ax.fill_between(x, y - std, y + std)

    ax.set_ylim(0.0, np.max(y + std) + 0.1)

    plt.tight_layout()
    fig.savefig(save_path)