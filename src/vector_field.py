import matplotlib.pyplot as plt
import numpy as np
import cmath


def loss(x, y):
    return x * y


def grad(x, y):
    return -y, x


def unrolled_loss(x, y, length_x, length_y):
    dy, dx = grad(x, y)
    # (x - length*dx)*(y - length*dy) = (x - length*(-y))*(y - length*(x))
    # (x + length*(y))*(y - length*(x)) = x*y - length*x*x + length*y*y - length*length*dx*dy
    return x*y + length_y*y*y - length_x*x*x - length_y*length_x*y*x


def unrolled_grad(x, y, length):
    dx = -(y - 2*length*x + length*length*y)
    dy = x - length*length*x + 2*length*y
    return dx, dy

def unrolled_finite_diff_grad(x, y, length):
    finite_size = 0.1 #1.0
    dx = -(y - (loss(x, y + finite_size) - loss(x, y - finite_size))/(2*finite_size))
    dy = x - (-loss(x + finite_size, y) - -loss(x - finite_size, y))/(2*finite_size)
    return dx, dy



def update(x, y, length, use_extrapolate, use_unrolled):
    dx, dy = grad(x, y)
    if use_extrapolate:
        dx, dy = grad(x - length*dx, y - length*dy)
        if use_unrolled:
            raise Exception
    elif use_unrolled:
        dx, dy = unrolled_grad(x, y, length)

    step_x = length * dx
    step_y = length * dy
    return step_x, step_y


def plot_vector_field(title, use_extrapolate=False, use_alternating=False, use_unrolling=False, do_xlabel=True, do_ylabel=True):
    fig_width, fig_height = 10, 10
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    num_rows, num_cols = 1, 1
    axs = [fig.add_subplot(num_rows, num_cols, ind + 1, frameon=False) for ind in range(num_rows * num_cols)]

    alpha = 0.15  # 0.1  # Learning rate

    num_steps = 50
    grid_density = 7
    grid_radius = 2.0
    color = 'red'
    for field_key_x, init_x in enumerate(np.linspace(-grid_radius, grid_radius, grid_density)):
        for field_key_y, init_y in enumerate(np.linspace(-grid_radius, grid_radius, grid_density)):
            x, y = init_x, init_y  # init_x + np.random.randn() * 0.1, init_y + np.random.randn() * 0.1
            results_x, results_y = [x], [y]
            losses = []
            for step in range(num_steps):
                dx, dy = update(x, y, alpha, use_extrapolate, use_unrolling)

                if use_alternating:
                    if step % 2 == 0:
                        x -= dx
                    else:
                        y -= dy
                else:
                    x -= dx
                    y -= dy

                results_x += [x]
                results_y += [y]
                losses += [loss(x, y)]

                # print(f"grad_x: {dx}, grad_y: {dy}")
                # print(f"Location: {x}, {y}, with loss = {losses[-1]}")

            for i in range(len(results_x) - 1):
                arrow_label = ''
                if field_key_x == 0 and field_key_y == 0:
                    arrow_label = 'Test'
                axs[0].arrow(x=results_x[i], y=results_y[i],
                             dx=results_x[i + 1] - results_x[i],
                             dy=results_y[i + 1] - results_y[i],
                             color=color, head_width=0.02, label=arrow_label)
            axs[0].plot(results_x, results_y, color=color, label=arrow_label)
            #if field_key_x == 0 and field_key_y == 0:
            #    axs[0].legend()

            # axs[1].plot(range(len(losses)), losses, label=label, c=colors[color_key])
    '''if do_xlabel:
        axs[0].set(xlabel='x')
    else:
        axs[0].set_xticks([])
    if do_ylabel:
        axs[0].set(ylabel='y')
    else:
        axs[0].set_yticks([])'''
    axs[0].set(xlabel='x')
    axs[0].set_xticks([-2, 0, 2])
    axs[0].set(ylabel='y')
    axs[0].set_yticks([-2, 0, 2])
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False)
    axs[0].grid(False)


    multiple = 1
    axs[0].set_xlim(-grid_radius * multiple, grid_radius * multiple)
    axs[0].set_ylim(-grid_radius * multiple, grid_radius * multiple)
    fig.savefig(title + ".pdf", bbox_inches='tight')
    plt.close('all')

    # TODO: Draw vector field at each location?  Maybe a graph with lines at each x, y in a grid
    # TODO: Solve for beta given eig and alpha
    # TODO: Simple estimate of complex curvature?
    # TODO: Randomly sample small boxes of params from each player which can be inverted?


# TODO: Document functions
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # Set some parameters.
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    font_size = 50
    mpl.rcParams['legend.fontsize'] = font_size
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size

    im_dir = 'images/'
    plot_vector_field(im_dir + 'base_simultaneous', do_xlabel=False, do_ylabel=False)
    plot_vector_field(im_dir + 'base_alternating', use_alternating=True, do_xlabel=False)
    plot_vector_field(im_dir + 'extrapolation_simultaneous', use_extrapolate=True, do_xlabel=False, do_ylabel=False)
    plot_vector_field(im_dir + 'extrapolation_alternating', use_extrapolate=True, use_alternating=True, do_xlabel=False)
    plot_vector_field(im_dir + 'unrolled_simultaneous', use_unrolling=True, do_ylabel=False)
    plot_vector_field(im_dir + 'unrolled_alternating', use_unrolling=True, use_alternating=True)