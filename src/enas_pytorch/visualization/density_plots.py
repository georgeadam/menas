import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def density_plot(values, title, x_label, save_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    sns.distplot(values, ax=ax)

    fig.suptitle(title)
    fig.savefig(save_path)