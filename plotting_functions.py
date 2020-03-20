import matplotlib.pyplot as plt
import numpy as np


def plot(data, title='', axis=None):
    if axis is None:
        fig, plot_axis = plt.subplots(1)
    else:
        plot_axis = axis
    plot_axis.plot(data)
    plot_axis.set_title(title)

    if axis is None:
        plt.show()
    else:
        return plot_axis


def imshow(data, title='', axis=None):
    if axis is None:
        fig, plot_axis = plt.subplots(1, figsize=(10, 8))
    else:
        plot_axis = axis
    pl = plot_axis.imshow(data, aspect='auto')
    plot_axis.set_title(title)
    plt.colorbar(pl, ax=plot_axis)

    if axis is None:
        plt.show()
    else:
        return plot_axis


def plot_alpha_channel(data, channel, axis=None):
    if axis is None:
        fig, plot_axis = plt.subplots(1)
    else:
        plot_axis = axis
    plot_axis.plot((data[:, channel]))
    plot_axis.set_xlim([0, 500])
    plot_axis.set_title(f'Source {channel}')

    if axis is None:
        plt.show()
    else:
        return plot_axis


def plot_multiple_channels(data, channels, axis=None):
    if axis is None:
        fig, plot_axis = plt.subplots(1)
    else:
        plot_axis = axis
    for channel in channels:
        plot_axis.plot((data[:, channel]), label=channel)
    plot_axis.set_xlim([0, 500])
    plot_axis.set_title(f'Sources {", ".join(map(str, channels))}')
    plot_axis.legend()

    if axis is None:
        plt.show()
    else:
        return plot_axis


def plot_reg(data, npriors, axis=None):
    if axis is None:
        fig, plot_axis = plt.subplots(1)
    else:
        plot_axis = axis
    plot_axis.plot((data[:, npriors - 1]))
    plot_axis.set_xlim([0, 500])
    plot_axis.set_title('reg')

    if axis is None:
        plt.show()
    else:
        return plot_axis


def plot_all(loss, trial_times, mu_rs, n_priors):
    print("Plotting results")
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    axis_generator = (axis for axis in axes.flatten())

    plot(np.log(loss - loss.min() + 1e-6), title="adjusted log loss", axis=next(axis_generator))

    imshow(trial_times[1:1000, :].T, title="trial_times", axis=next(axis_generator))
    imshow(mu_rs[1:1000, 88:100].T, title="Recon alphas", axis=next(axis_generator))
    imshow(mu_rs[1:1000, :].T, title="Recon alphas", axis=next(axis_generator))

    plot_alpha_channel(mu_rs, channel=88, axis=next(axis_generator))
    plot_alpha_channel(mu_rs, channel=89, axis=next(axis_generator))
    plot_alpha_channel(mu_rs, channel=98, axis=next(axis_generator))

    plot_multiple_channels(mu_rs, channels=[89, 98], axis=next(axis_generator))

    plot_reg(mu_rs, npriors=n_priors, axis=next(axis_generator))

    plt.tight_layout()

    plt.show()
