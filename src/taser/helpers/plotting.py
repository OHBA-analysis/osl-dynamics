"""Helper functions for plotting using matplotlib

"""
import logging
from itertools import zip_longest

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from taser.helpers.array_ops import (
    from_cholesky,
    get_one_hot,
    reduce_state_time_course,
    state_activation,
    state_lifetimes,
)
from taser.helpers.decorators import transpose
from taser.helpers.misc import override_dict_defaults


@transpose(0, "time_series")
def value_separation(
    time_series: np.ndarray, separation_factor: float = 1.2
) -> np.ndarray:
    gap = separation_factor * np.abs([time_series.min(), time_series.max()]).max()
    separation = np.arange(time_series.shape[1])[None, ::-1]
    return time_series + gap * separation


def get_colors(n_states: int, colormap: str = "gist_rainbow"):
    colormap = plt.get_cmap(colormap)
    colors = [colormap(1 * i / n_states) for i in range(n_states)]
    return colors


@transpose(0, 1, 2, "time_series_0", "time_series_1", "time_series_2")
def plot_two_data_scales(
    time_series_0: np.ndarray,
    time_series_1: np.ndarray,
    time_series_2: np.ndarray = None,
    n_time_points: int = np.inf,
    fig_kwargs: dict = None,
    plot_0_kwargs: dict = None,
    plot_1_kwargs: dict = None,
    plot_2_kwargs: dict = None,
):
    n_time_points = min(n_time_points, time_series_0.shape[0], time_series_1.shape[0],)

    if plot_2_kwargs is not None:
        n_time_points = min(n_time_points, time_series_2.shape[0])

    fig_defaults = {"figsize": (20, 10), "sharex": "all"}
    fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
    fig, axes = plt.subplots(2, **fig_kwargs)

    plot_0_defaults = {"lw": 0.6, "color": "tab:blue"}
    plot_0_kwargs = override_dict_defaults(plot_0_defaults, plot_0_kwargs)
    axes[0].plot(value_separation(time_series_0[:n_time_points]), **plot_0_kwargs)

    plot_1_defaults = {"lw": 0.6, "color": "tab:blue"}
    plot_1_kwargs = override_dict_defaults(plot_1_defaults, plot_1_kwargs)
    axes[1].plot(value_separation(time_series_1[:n_time_points]), **plot_1_kwargs)

    if time_series_2 is not None:
        plot_2_defaults = {"lw": 0.4, "color": "tab:orange"}
        plot_2_kwargs = override_dict_defaults(plot_2_defaults, plot_2_kwargs)
        axes[1].plot(value_separation(time_series_1[:n_time_points]), **plot_2_kwargs)

    for axis in axes:
        axis.autoscale(axis="x", tight=True)
        axis.set_yticks([])

    plt.tight_layout()

    plt.show()


@transpose(0, 1, "time_series", "state_time_course")
def plot_state_highlighted_data(
    time_series: np.ndarray,
    state_time_course: np.ndarray,
    events: np.ndarray = None,
    n_time_points: int = 5000,
    colormap: str = "gist_rainbow",
    fig_kwargs: dict = None,
    highlight_kwargs: dict = None,
    plot_kwargs: dict = None,
    event_kwargs: dict = None,
    file_name: str = None,
):
    fig_defaults = {
        "figsize": (20, 10),
        "gridspec_kw": {"height_ratios": [1] if events is None else [1, 5]},
        "sharex": "all",
    }
    event_defaults = {"color": "tab:blue"}
    fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
    event_kwargs = override_dict_defaults(event_defaults, event_kwargs)

    fig, axes = plt.subplots(1 if events is None else 2, **fig_kwargs, squeeze=False)

    axes = axes.ravel()

    plot_time_series(
        time_series, axes[-1], n_time_points=n_time_points, plot_kwargs=plot_kwargs
    )
    highlight_states(
        state_time_course,
        axes[-1],
        n_time_points=n_time_points,
        colormap=colormap,
        highlight_kwargs=highlight_kwargs,
    )

    if events is not None:
        axes[0].plot(events[:n_time_points], **event_kwargs)

    for axis in axes:
        axis.autoscale(tight=True)
        axis.axis("off")

    plt.tight_layout()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, dpi=350)
        plt.close("all")


@transpose("time_series", 0)
def plot_time_series(
    time_series: np.ndarray,
    axis: plt.Axes = None,
    n_time_points: int = 5000,
    plot_kwargs: dict = None,
    fig_kwargs: dict = None,
    y_tick_values: list = None,
):
    n_time_points = min(n_time_points, time_series.shape[0])
    axis_given = axis is not None
    if not axis_given:
        fig_defaults = {
            "figsize": (20, 10),
        }
        fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
        fig, axis = plt.subplots(1, **fig_kwargs)

    default_plot_kwargs = {
        "lw": 0.7,
        "color": "tab:blue",
    }
    plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    n_channels = time_series.shape[1]

    separation = (
        np.maximum(time_series[:n_time_points].max(), time_series[:n_time_points].min())
        * 1.2
    )
    gaps = np.arange(n_channels)[::-1] * separation

    axis.plot(time_series[:n_time_points] + gaps[None, :], **plot_kwargs)

    axis.autoscale(tight=True)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_xticks([])

    if y_tick_values is not None:
        axis.set_yticks(gaps)
        axis.set_yticklabels(y_tick_values)
    else:
        axis.set_yticks([])

    if not axis_given:
        plt.tight_layout()
        plt.show()


@transpose(0, "state_time_course")
def highlight_states(
    state_time_course: np.ndarray,
    axis: plt.Axes = None,
    colormap: str = "gist_rainbow",
    n_time_points: int = 5000,
    highlight_kwargs: dict = None,
    legend: bool = True,
    fig_kwargs: dict = None,
):
    n_time_points = min(n_time_points, state_time_course.shape[0])
    axis_given = axis is not None
    if not axis_given:
        fig_defaults = {
            "figsize": (20, 3),
        }
        fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
        fig, axis = plt.subplots(1, **fig_kwargs)

    highlight_defaults = {"alpha": 0.2, "lw": 0}
    highlight_kwargs = override_dict_defaults(highlight_defaults, highlight_kwargs)

    reduced_state_time_course = reduce_state_time_course(state_time_course)
    n_states = reduced_state_time_course.shape[1]
    ons, offs = state_activation(reduced_state_time_course)

    colors = get_colors(n_states, colormap)

    for state_number, (state_ons, state_offs, color) in enumerate(
        zip(ons, offs, colors)
    ):
        for highlight_number, (on, off) in enumerate(
            zip(state_ons[:n_time_points], state_offs[:n_time_points])
        ):
            if (on > n_time_points) and (off > n_time_points):
                break
            handles, labels = axis.get_legend_handles_labels()
            axis.axvspan(
                on,
                min(off, n_time_points),
                color=color,
                **highlight_kwargs,
                label=str(state_number)
                if (str(state_number) not in labels) and legend
                else "",
            )

    if legend:
        axis.legend(
            loc=(0.0, -0.2), mode="expand", borderaxespad=0, ncol=n_states,
        )

    axis.set_yticks([])

    axis.autoscale(tight=True)
    axis.set_xlim(0, n_time_points)

    plt.tight_layout()

    if not axis_given:
        axis.axis("off")
        plt.show()


def plot_cholesky(matrix, group_color_scale: bool = True):
    matrix = np.array(matrix)
    c_i = from_cholesky(matrix)
    plot_matrices(c_i, group_color_scale=group_color_scale)


def plot_matrix_max_min_mean(
    matrix, group_color_scale: bool = True, cholesky: bool = False
):
    if cholesky:
        matrix = from_cholesky(matrix)
    element_max = matrix.max(axis=0)
    element_min = matrix.min(axis=0)
    element_mean = matrix.mean(axis=0)
    plot_matrices(
        [element_max, element_min, element_mean],
        group_color_scale=group_color_scale,
        titles=["max", "min", "mean"],
    )


def plot_matrices(matrix, group_color_scale: bool = True, titles: list = None):
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, :]
    if matrix.ndim != 3:
        raise ValueError("Must be a 3D array.")
    short, long, empty = rough_square_axes(len(matrix))
    if group_color_scale:
        v_min = matrix.min()
        v_max = matrix.max()
    f_width = 2.5 * short
    f_height = 2.5 * long
    fig, axes = plt.subplots(
        ncols=short, nrows=long, figsize=(f_width, f_height), squeeze=False
    )

    if titles is None:
        titles = [""] * len(matrix)

    for grid, axis, title in zip_longest(matrix, axes.ravel(), titles):
        if grid is None:
            axis.remove()
            continue
        if group_color_scale:
            im = axis.matshow(grid, vmin=v_min, vmax=v_max)
        else:
            im = axis.matshow(grid)
        axis.set_title(title)

    if group_color_scale:
        fig.subplots_adjust(right=0.8)
        color_bar_axis = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=color_bar_axis)
    else:
        for axis in fig.axes:
            pl = axis.get_images()[0]
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pl, cax=cax)
        plt.tight_layout()

    plt.show()


def rough_square_axes(n_plots):
    short = np.floor(n_plots ** 0.5).astype(int)
    long = np.ceil(n_plots ** 0.5).astype(int)
    if short * long < n_plots:
        short += 1
    empty = short * long - n_plots
    return short, long, empty


def add_axis_colorbar(axis: plt.Axes):
    try:
        pl = axis.get_images()[0]
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pl, cax=cax)
    except IndexError:
        logging.warning("No mappable image found on axis.")


@transpose(0, "state_time_course")
def plot_state_lifetimes(
    state_time_course: np.ndarray,
    figsize=(5, 5),
    bins: int = 20,
    density: bool = False,
    hist_kwargs: dict = None,
):
    if state_time_course.ndim == 1:
        state_time_course = get_one_hot(state_time_course)
    if state_time_course.ndim != 2:
        raise ValueError("state_timecourse must be a 2D array")

    default_hist_kwargs = {"alpha": 0.5}
    hist_kwargs = override_dict_defaults(default_hist_kwargs, hist_kwargs)

    state_time_course = reduce_state_time_course(state_time_course)
    channel_lifetimes = state_lifetimes(state_time_course)
    n_plots = state_time_course.shape[1]
    short, long, empty = rough_square_axes(n_plots)

    colors = get_colors(n_plots)

    fig, axes = plt.subplots(short, long, figsize=figsize)
    for channel, axis, color in zip_longest(channel_lifetimes, axes.ravel(), colors):
        if channel is None:
            axis.remove()
            continue
        axis.hist(channel, density=density, bins=bins, color=color, **hist_kwargs)
        t = axis.text(
            0.95,
            0.95,
            f"{np.sum(channel) / len(state_time_course) * 100:.2f}%",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
        )
        t.set_bbox({"facecolor": "white", "alpha": 0.7, "boxstyle": "round"})
    plt.tight_layout()

    plt.show()


@transpose(0, 1, "time_series", "state_time_course")
def plot_highlighted_states(
    time_series, state_time_course, n_time_points, state_order, fig_kwargs=None
):
    if fig_kwargs is None:
        fig_kwargs = {}

    fig, axes = plt.subplots(state_time_course.shape[1], **fig_kwargs)

    ons, offs = state_activation(state_time_course)

    for i, (channel_ons, channel_offs, ss, axis) in enumerate(
        zip(ons, offs, state_order, axes.ravel())
    ):
        axis.plot(time_series[0:n_time_points, ss])
        axis.set_yticks([0.5])
        axis.set_yticklabels([i])
        axis.set_ylim(-0.1, 1.1)

        axis_highlights(channel_ons, channel_offs, n_time_points, axis)
        axis.autoscale(axis="x", tight=True)

    plt.show()


def axis_highlights(ons, offs, n_points, axis, i: int = 0, label: str = ""):
    for on, off in zip(ons, offs):
        if on < n_points or off < n_points:
            axis.axvspan(on, min(n_points - 1, off), alpha=0.1, color="r")
