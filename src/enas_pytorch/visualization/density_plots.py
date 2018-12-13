import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import text
import seaborn as sns
import numpy as np
sns.set()

left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height


def density_plot(values, title, x_label, save_path, label_stats=True):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    sns.distplot(values, ax=ax)

    if label_stats:
        ax.text(right - 0.02, top - 0.02,
                "Mean: {:.2f} \nMin: {:.2f}\nMax: {:.2f}".format(np.mean(values),
                                                               np.min(values),
                                                               np.max(values)),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes)

    fig.suptitle(title)
    fig.savefig(save_path)