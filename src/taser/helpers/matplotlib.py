"""Helper functions for plotting using matplotlib

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_two_data_scales(
    dataset_0: np.ndarray,
    dataset_1: np.ndarray,
    dataset_2: np.ndarray = None,
    dataset_0_cap: float = None,
    dataset_1_cap: float = None,
    dataset_2_cap: float = None,
    n_points: int = None,
):
    """Plot data with same x values on two y-axis scales.

    Given two (or three) multi-channel datasets with two different y-scales, create a
    single figure which contains multiple subplots. Each dataset will produce a
    collection of subplots with equal y-scaling. All subplots share the same x-scale.
    If a third dataset is provided, it will be overlaid on the second dataset.

    Caps can be provided which clip each dataset to a maximum value.

    Parameters
    ----------
    dataset_0 : numpy.ndarray
        Dataset to be plotted individually (e.g. raw signal). Should be provided
        with dimensions [channels x samples]. Each channel will be plotted in a
        separate subplot.
    dataset_1 : numpy.ndarray
        Dataset to be plotted either individually or with `dataset_2`. Should be provided
        with dimensions [channels x samples]. Each channel will be plotted in a separate
        subplot.
    dataset_2 : numpy.ndarray
        Dataset to be plotted with `dataset_2`. Should be provided with dimensions
        [channels x samples]. Each channel will be overlaid on separate subplots for
        dataset_1.
    dataset_0_cap : float
        All values above `dataset_0_cap` in `dataset_0` will be clipped. Default is no
        clipping.
    dataset_1_cap : float
        All values above `dataset_1_cap` in `dataset_1` will be clipped. Default is no
        clipping.
    dataset_2_cap : float
        All values above `dataset_2_cap` in `dataset_2` will be clipped. Default is no
        clipping.
    n_points : int
        Number of sample points to plot.
    """
    if dataset_1.shape != dataset_2.shape:
        raise ValueError("dataset_1 and dataset_2 must have the same 2D shape")

    if dataset_0_cap is not None:
        dataset_0[dataset_0 > dataset_0_cap] = dataset_0_cap
    if dataset_1_cap is not None:
        dataset_1[dataset_1 > dataset_1_cap] = dataset_1_cap
    if dataset_2_cap is not None:
        dataset_2[dataset_2 > dataset_2_cap] = dataset_2_cap

    if n_points is None:
        n_points = dataset_0.shape[1]

    n_channels = dataset_0.shape[0]
    n_states = dataset_1.shape[0]

    fig, axes = plt.subplots(
        n_channels + n_states, figsize=(20, 10), sharex="all", sharey="none"
    )

    for d0, axis in zip(dataset_0, axes[:n_channels]):
        axis.plot(d0[:n_points])
    for d1, axis in zip(dataset_1, axes[n_channels:]):
        axis.plot(d1[:n_points])
    if dataset_2 is not None:
        for d2, axis in zip(dataset_2, axes[n_channels:]):
            axis.plot(d2[:n_points])

    y_limits = [axis.get_ylim() for axis in axes[:n_channels]]
    y_min = min(y_limits)[0]
    y_max = max(y_limits[1])
    for axis in axes[:n_channels]:
        axis.set_ylim(y_min, y_max)

    y_limits = [axis.get_ylim() for axis in axes[n_channels:]]
    y_min = min(y_limits)[0]
    y_max = max(y_limits[1])
    for axis in axes[n_channels:]:
        axis.set_ylim(y_min, y_max)

    plt.show()


def plot_covariances(cholesky_djs: np.ndarray, fig_kwargs=None):
    """Function for plotting multiple matrices in a row.

    Parameters
    ----------
    cholesky_djs : numpy.array
        Cholesky decomposition of state covariances [n_states x n_channels x n_channels]
    fig_kwargs
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    c_i = cholesky_djs @ cholesky_djs.transpose((0, 2, 1))
    fig, axes = plt.subplots(ncols=len(c_i), **fig_kwargs)
    for covariance, axis in zip(c_i, axes):
        pl = axis.matshow(covariance)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pl, cax=cax)
    plt.show()


def time_course_plot_axis(one_hot_time_course, axis, n_plot_points, title=None):
    for state in one_hot_time_course:
        axis.fill_between(
            np.arange(n_plot_points), state[:n_plot_points], alpha=0.5, lw=10
        )
        axis.plot(np.arange(n_plot_points), state[:n_plot_points], lw=4, c="k")
    if title is not None:
        axis.set_title(title)
    axis.autoscale(tight=True)


def time_course_plot(one_hot_time_courses, n_plot_points, titles=None, figsize=(15, 6)):
    fig, axes = plt.subplots(nrows=len(one_hot_time_courses), figsize=figsize)

    if titles is None or (len(titles) != len(one_hot_time_courses)):
        titles = ["" for i in one_hot_time_courses]

    for one_hot_time_course, axis, title in zip(one_hot_time_courses, axes, titles):
        time_course_plot_axis(one_hot_time_course, axis, n_plot_points, title)
    plt.tight_layout()
    plt.show()
