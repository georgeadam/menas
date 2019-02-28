import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
sns.set()


def heatmap(data, title, save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    sns.heatmap(data, ax=ax, cmap=sns.color_palette("RdBu_r", 7))
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    fig.suptitle(title)
    fig.savefig(save_path)