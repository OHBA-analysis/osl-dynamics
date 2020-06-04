"""Helper functions for plotting using matplotlib

"""
import logging
from itertools import zip_longest
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from taser import array_ops

from taser.array_ops import (
    from_cholesky,
    get_one_hot,
    state_activation,
    state_lifetimes,
    match_states,
    correlate_states,
    mean_diagonal,
)
from taser.decorators import transpose
from taser.helpers.misc import override_dict_defaults


def plot_correlation(state_time_course_1: np.ndarray, state_time_course_2: np.ndarray):
    matched_stc_1, matched_stc_2 = match_states(
        state_time_course_1, state_time_course_2
    )
    correlation = correlate_states(matched_stc_1, matched_stc_2)
    correlation_off_diagonal = mean_diagonal(correlation)

    plot_matrices([correlation, correlation_off_diagonal], group_color_scale=False)


def plot_state_sums(state_time_course: np.ndarray, color="tab:gray"):
    fig, axis = plt.subplots(1)
    counts = state_time_course.sum(axis=0).astype(int)
    bars = axis.bar(range(len(counts)), counts, color=color)
    axis.set_xticks(range(len(counts)))
    text = []
    for i, val in enumerate(counts):
        text.append(
            axis.text(
                i,
                val * 0.98,
                val,
                horizontalalignment="center",
                verticalalignment="top",
                color="white",
                fontsize=13,
            )
        )
    plt.setp(axis.spines.values(), visible=False)
    plt.setp(axis.get_xticklines(), visible=False)
    axis.set_yticks([])

    adjust_text(text, bars, fig, axis)

    plt.show()


# noinspection PyUnresolvedReferences
def adjust_text(
    text_objects: List[matplotlib.text.Text],
    plot_objects: List[matplotlib.patches.Patch],
    fig=None,
    axis=None,
    color="black",
):
    fig = plt.gcf() if fig is None else fig
    axis = plt.gca() if axis is None else axis

    fig.canvas.draw()

    plot_bbs = [plot_object.get_bbox() for plot_object in plot_objects]
    text_bbs = [
        text_object.get_window_extent().inverse_transformed(axis.transData)
        for text_object in text_objects
    ]

    while not all(
        [
            plot_bb.containsx(text_bb.x0) and plot_bb.containsx(text_bb.x1)
            for plot_bb, text_bb in zip(plot_bbs, text_bbs)
        ]
    ):
        plot_bbs = [plot_object.get_bbox() for plot_object in plot_objects]
        text_bbs = [
            text_object.get_window_extent().inverse_transformed(axis.transData)
            for text_object in text_objects
        ]
        for plot_bb, text_bb, text_object in zip(plot_bbs, text_bbs, text_objects):
            if not (plot_bb.containsx(text_bb.x0) and plot_bb.containsx(text_bb.x1)):
                text_object.set_size(text_object.get_size() - 1)
        fig.canvas.draw()

    for plot_bb, text_bb, text_object in zip(plot_bbs, text_bbs, text_objects):
        if not (plot_bb.containsy(text_bb.y0) and plot_bb.containsy(text_bb.y1)):
            text_object.set_verticalalignment("bottom")
            text_object.set_y(plot_bb.y1)
            text_object.set_color(color)

    fig.canvas.draw()


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
    sample_frequency: float = 1,
    highlight_kwargs: dict = None,
    legend: bool = True,
    fig_kwargs: dict = None,
):
    if n_time_points is None:
        n_time_points = state_time_course.shape[0]
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

    # reduced_state_time_course = reduce_state_time_course(state_time_course)
    reduced_state_time_course = state_time_course.copy()
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
            if (str(state_number) not in labels) and legend:
                label = str(state_number)
            else:
                label = ""
            axis.axvspan(
                on / sample_frequency,
                min(off, n_time_points) / sample_frequency,
                color=color,
                **highlight_kwargs,
                label=label,
            )

    if legend:
        axis.legend(
            loc=(0.0, -0.3), mode="expand", borderaxespad=0, ncol=n_states,
        )

    plt.setp(axis.spines.values(), visible=False)

    axis.set_yticks([])

    axis.autoscale(tight=True)

    axis.set_xlim(0, n_time_points / sample_frequency)

    plt.tight_layout()

    if not axis_given:
        # axis.axis("off")
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


def plot_matrices(
    matrix,
    group_color_scale: bool = True,
    titles: list = None,
    cmap="viridis",
    nan_color="white",
):
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

    matplotlib.cm.get_cmap().set_bad(color=nan_color)

    for grid, axis, title in zip_longest(matrix, axes.ravel(), titles):
        if grid is None:
            axis.remove()
            continue
        if group_color_scale:
            im = axis.matshow(grid, vmin=v_min, vmax=v_max, cmap=cmap)
        else:
            im = axis.matshow(grid, cmap=cmap)
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
    long = np.floor(n_plots ** 0.5).astype(int)
    short = np.ceil(n_plots ** 0.5).astype(int)
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
    bins: int = 20,
    density: bool = False,
    match_scale_x=True,
    match_scale_y=True,
    hist_kwargs: dict = None,
    fig_kwargs: dict = None,
):
    if state_time_course.ndim == 1:
        state_time_course = get_one_hot(state_time_course)
    if state_time_course.ndim != 2:
        raise ValueError("state_timecourse must be a 2D array")

    # state_time_course = reduce_state_time_course(state_time_course)
    channel_lifetimes = state_lifetimes(state_time_course)
    n_plots = state_time_course.shape[1]
    short, long, empty = rough_square_axes(n_plots)

    colors = get_colors(n_plots)

    default_hist_kwargs = {"alpha": 0.5}
    hist_kwargs = override_dict_defaults(default_hist_kwargs, hist_kwargs)

    default_fig_kwargs = {"figsize": (long * 2.5, short * 2.5)}
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    fig, axes = plt.subplots(short, long, **fig_kwargs)
    largest_bar = 0
    furthest_value = 0
    for channel, axis, color in zip_longest(channel_lifetimes, axes.ravel(), colors):
        if channel is None:
            axis.remove()
            continue
        if not len(channel):
            # axis.hist([])
            axis.text(
                0.5,
                0.5,
                "No\nactivation",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
                fontsize=20,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        hist = axis.hist(
            channel, density=density, bins=bins, color=color, **hist_kwargs
        )
        largest_bar = max(hist[0].max(), largest_bar)
        furthest_value = max(hist[1].max(), furthest_value)
        t = axis.text(
            0.95,
            0.95,
            f"{np.sum(channel) / len(state_time_course) * 100:.2f}%",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
        )
        axis.xaxis.set_tick_params(labelbottom=True, labelleft=True)
        t.set_bbox({"facecolor": "white", "alpha": 0.7, "boxstyle": "round"})
    for axis in axes.ravel():
        if match_scale_x:
            axis.set_xlim(0, furthest_value * 1.1)
        if match_scale_y:
            axis.set_ylim(0, largest_bar * 1.1)
    plt.tight_layout()

    plt.show()


@transpose(0, 1, "time_series", "state_time_course")
def plot_highlighted_states(
    time_series, state_time_course, n_time_points, fig_kwargs=None
):
    if fig_kwargs is None:
        fig_kwargs = {}

    fig, axes = plt.subplots(state_time_course.shape[1], **fig_kwargs)

    ons, offs = state_activation(state_time_course)

    for i, (channel_ons, channel_offs, axis, channel_time_series) in enumerate(
        zip(ons, offs, axes.ravel(), time_series[:n_time_points].T)
    ):
        axis.plot(channel_time_series)
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


def compare_state_data(
    *state_time_courses, n_time_points=20000, sample_frequency: float = 1,
):
    fig, axes = plt.subplots(
        nrows=len(state_time_courses),
        figsize=(20, 2.5 * len(state_time_courses)),
        sharex="all",
    )

    for state_time_course, axis in zip(state_time_courses[:-1], axes[:-1]):
        highlight_states(
            state_time_course,
            axis=axis,
            n_time_points=n_time_points,
            legend=False,
            sample_frequency=sample_frequency,
        )

    highlight_states(
        state_time_courses[-1],
        axis=axes[-1],
        n_time_points=n_time_points,
        legend=True,
        sample_frequency=sample_frequency,
    )


@transpose("state_time_course_1", 0, "state_time_course_2", 1)
def confusion_matrix(state_time_course_1: np.ndarray, state_time_course_2: np.ndarray):
    confusion = array_ops.confusion_matrix(state_time_course_1, state_time_course_2)
    nan_diagonal = confusion.copy().astype(float)
    nan_diagonal[np.diag_indices_from(nan_diagonal)] = np.nan
    plot_matrices([confusion, nan_diagonal], group_color_scale=False)
