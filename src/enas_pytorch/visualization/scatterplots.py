import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
sns.set()

from matplotlib.ticker import NullFormatter


def scatterplot(Ys, titles, colors, save_path):
    (fig, subplots) = plt.subplots(2, 2, figsize=(20, 20))

    for i in range(len(Ys)):
        ax = subplots[i // 2][i % 2]
        # ax = fig.add_subplot(len(Ys), 1, i+1)

        ax.set_title(titles[i])
        ax.scatter(Ys[i][:, 0], Ys[i][:, 1], c=colors[i])
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    fig.savefig(save_path)