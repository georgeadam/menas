import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def line_plot(x, y, title, x_label, y_label, save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    for i in range(len(x)):
        sns.lineplot(x[i], y[i], ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.suptitle(title)

    fig.savefig(save_path)