"""Helper functions for plotting using matplotlib

"""
import logging
from itertools import zip_longest
from typing import Any, Iterable, List, Tuple, Union

import matplotlib
import numpy as np
import vrad.inference.metrics
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vrad.array_ops import (
    correlate_states,
    from_cholesky,
    get_one_hot,
    match_states,
    mean_diagonal,
    state_activation,
    state_lifetimes,
)
from vrad.utils.decorators import deprecated, transpose
from vrad.utils.misc import override_dict_defaults

_logger = logging.getLogger("VRAD")


def plot_correlation(
    state_time_course_1: np.ndarray,
    state_time_course_2: np.ndarray,
    show_diagonal: bool = True,
):
    """Plot a correlation matrix between two state time courses.

    Given two state time courses, find the correlation between the states and
    show both the full correlation matrix, and the correlation matrix with diagonals
    removed. This behaviour can be modified using the show_diagonal parameter.

    Parameters
    ----------
    state_time_course_1 : numpy.ndarray
        First state time course (one-hot).
    state_time_course_2 : numpy.ndarray
        Second state time course (one-hot).
    show_diagonal : bool
        If True (default), the correlation matrix with the diagonal removed will also
        be shown.
    """
    matched_stc_1, matched_stc_2 = match_states(
        state_time_course_1, state_time_course_2
    )
    correlation = correlate_states(matched_stc_1, matched_stc_2)
    correlation_off_diagonal = mean_diagonal(correlation)

    if show_diagonal:
        plot_matrices([correlation, correlation_off_diagonal], group_color_scale=False)
    else:
        plot_matrices([correlation], group_color_scale=False)


def plot_state_sums(
    state_time_course: np.ndarray, color: str = "tab:gray", filename: str = None,
):
    """Bar chart of total state durations.

    Creates a bar chart of the total activation duration for each state in
    state_time_course. The value of each bar is displayed at the top of each bar.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        The state time course to be plotted (one-hot).
    color : str
        A matplotlib compatible color given as a string. Default is "tab:gray".
    filename: str
        A file to which to save the figure.
    """
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

    show_or_save(filename)


# noinspection PyUnresolvedReferences
def adjust_text(
    text_objects: List[matplotlib.text.Text],
    plot_objects: List[matplotlib.patches.Patch],
    fig: matplotlib.figure.Figure = None,
    axis: matplotlib.axes.Axes = None,
    color: str = "black",
):
    """Make text fit in bar charts.

    Given a list of bars and a list of text objects to fit into them, find the largest
    text size which fits within the each bar. If the text is too tall to fit within the
    bar, it will be placed directly above the bar.

    Parameters
    ----------
    text_objects : list of matplotlib.text.Text
        The text to be placed within the bars.
    plot_objects : list of matplotlib.patches.Patch
        The bars to place the text within.
    fig : matplotlib.figure.Figure
        Figure to work on. Default is to get the current active figure.
    axis : matplotlib.axes.Axes
        Axis to work on. Default is to get the current active axis.
    color: str
        Color for the text as a matplotlib compatible string. Default black.
    """
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
    """Separate sequences for plotting.

    Convenience method to add a constant to each channel of time_series. This allows
    easy plotting of multiple channels on the same axes with separation between them.
    The separation is determined by finding the largest and smallest value in any
    sequence and multiplying it by separation_factor.

    Parameters
    ----------
    time_series : numpy.ndarray
        The time series to be modified.
    separation_factor : float
        The factor by which to multiply the separation distance.

    Returns
    -------
    separated_time_series: numpy.ndarray
        The time series with separation added.

    """
    gap = separation_factor * np.abs([time_series.min(), time_series.max()]).max()
    separation = np.arange(time_series.shape[1])[None, ::-1]
    return time_series + gap * separation


def get_colors(
    n_states: int, colormap: str = "gist_rainbow"
) -> List[Tuple[float, float, float, float]]:
    """Produce equidistant colors from a matplotlib colormap.

    Given a matplotlib colormap, produce a series of RGBA colors which are equally
    spaced by value. There is no guarantee that these colors will be perceptually
    uniformly distributed and with many colors will likely be extremely close. Alpha is
    1.0 for all colors.

    Parameters
    ----------
    n_states : int
        The number of colors to return.
    colormap : str
        The name of a matplotlib colormap.

    Returns
    -------
    colors: list of tuple of float
        A list of colors in RGBA format. A = 1.0 in all cases.
    """
    colormap = plt.get_cmap(colormap)
    colors = [colormap(1 * i / n_states) for i in range(n_states)]
    return colors


@transpose(0, 1, 2, "time_series_0", "time_series_1", "time_series_2")
def plot_two_data_scales(
    time_series_0: np.ndarray,
    time_series_1: np.ndarray,
    time_series_2: np.ndarray = None,
    n_time_points: int = None,
    fig_kwargs: dict = None,
    plot_0_kwargs: dict = None,
    plot_1_kwargs: dict = None,
    plot_2_kwargs: dict = None,
    filename: str = None,
):
    n_time_points = min(
        n_time_points or np.inf, time_series_0.shape[0], time_series_1.shape[0],
    )

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

    show_or_save(filename)


@transpose(0, 1, "time_series", "state_time_course")
def plot_state_highlighted_data(
    time_series: Union[np.ndarray, Any],
    state_time_course: np.ndarray,
    events: np.ndarray = None,
    n_time_points: int = None,
    colormap: str = "gist_rainbow",
    fig_kwargs: dict = None,
    highlight_kwargs: dict = None,
    plot_kwargs: dict = None,
    event_kwargs: dict = None,
    filename: str = None,
):
    """Plot time series data highlighted by state.

    Plot a time series and highlight it by the active state defined
    by state_time_course. When working with task data, the events can be plotted above
    using the events option.

    Parameters
    ----------
    time_series: numpy.ndarray
        The data to be plotted.
    state_time_course: numpy.ndarray
        The states to highlight the data with.
    events: numpy.ndarray
        Optional. Events as a time series.
    n_time_points: int
        Number of time points to be plotted.
    colormap: str
        A matplotlib compatible colormap given as a string. This is for the highlights.
    fig_kwargs: dict
        A dictionary of kwargs to be passed to matplotlib.pyplot.subplots
    highlight_kwargs: dict
        A dictionary of kwargs for the highlights to be
        passed to matplotlib.pyplot.axvline.
    plot_kwargs: dict
        A dictionary of kwargs for the plotted data to be
        passed to matplotlib.pyplot.plot.
    event_kwargs: dict
        A dictionary of kwargs for the plotted event data to be
        passed to matplotlib.pyplot.plot
    filename: str
        A file to which to save the figure.
    """
    fig_defaults = {
        "figsize": (20, 10),
        "gridspec_kw": {"height_ratios": [1] if events is None else [1, 5]},
        "sharex": "all",
    }
    event_defaults = {"color": "tab:blue"}
    fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
    event_kwargs = override_dict_defaults(event_defaults, event_kwargs)

    n_time_points = min(
        n_time_points or np.inf, state_time_course.shape[0], time_series.shape[0]
    )

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

    show_or_save(filename)


@transpose("time_series", 0)
def plot_time_series(
    time_series: np.ndarray,
    axis: plt.Axes = None,
    n_time_points: int = None,
    plot_kwargs: dict = None,
    fig_kwargs: dict = None,
    y_tick_values: list = None,
    filename: str = None,
):
    """Plot a time series with channel separation.

    Plot time_series with channel separation calculated by
    vrad.plotting.value_separation. This allows each channel to be plotted on the
    same axis but with a separation from the other channels defined by their extreme
    values.

    Parameters
    ----------
    time_series: numpy.ndarray
        The time series to be plotted.
    axis: matplotlib.axes.Axes
        The axis on which to plot the data. If not given, a new axis is created.
    n_time_points: int
        The number of time points to be plotted.
    plot_kwargs: dict
        Keyword arguments to be passed on to matplotlib.pyplot.plot.
    fig_kwargs: dict
        Keyword arguments to be passed on to matplotlib.pyplot.subplots.
    y_tick_values:
        Labels for the channels to be placed on the y-axis.
    filename: str
        A file to which to save the figure.
    """
    n_time_points = min(n_time_points or np.inf, time_series.shape[0])
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
        show_or_save(filename)


@transpose("state_time_course", 0)
def state_barcode(
    state_time_course: np.ndarray,
    axis: plt.Axes = None,
    colormap: str = "gist_rainbow",
    n_time_points: int = None,
    sample_frequency: float = 1,
    highlight_kwargs: dict = None,
    legend: bool = True,
    fig_kwargs: dict = None,
    filename: str = None,
):
    state_time_course = np.asarray(state_time_course)
    if state_time_course.ndim == 2:
        state_time_course = state_time_course.argmax(axis=1)
    if state_time_course.ndim != 1:
        raise ValueError("state_time_course must be 1D or 2D.")

    n_time_points = min(n_time_points or np.inf, len(state_time_course))
    state_time_course = state_time_course[:n_time_points]

    axis_given = axis is not None
    if not axis_given:
        fig_defaults = {
            "figsize": (24, 2.5),
        }
        fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
        fig, axis = plt.subplots(1, **fig_kwargs)

    highlight_defaults = {"alpha": 0.2}
    highlight_kwargs = override_dict_defaults(highlight_defaults, highlight_kwargs)

    n_states = np.max(np.unique(state_time_course)) + 1

    cmap = plt.cm.get_cmap(colormap, lut=n_states)

    axis.imshow(
        state_time_course[None],
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=np.max(n_states) - 0.5,
        interpolation="none",
        extent=[0, n_time_points / sample_frequency, 0, 1],
        **highlight_kwargs,
    )

    plt.setp(axis.spines.values(), visible=False)

    if legend:
        add_axis_colorbar(axis)

    axis.set_yticks([])

    if not axis_given:
        show_or_save(filename)


@deprecated(replaced_by="state_barcode", reason="state_barcode is more efficient.")
@transpose(0, "state_time_course")
def highlight_states(
    state_time_course: np.ndarray,
    axis: plt.Axes = None,
    colormap: str = "gist_rainbow",
    n_time_points: int = None,
    sample_frequency: float = 1,
    highlight_kwargs: dict = None,
    legend: bool = True,
    fig_kwargs: dict = None,
    filename: str = None,
):
    """Plot vertical bars corresponding to state activation.

    For a state time course, each state has its activation starts and stops calculated
    using vrad.array_ops.state_activation. Each state is represented by a color chosen
    at uniform separations from the colormap given. If a sample frequency is provided,
    the x axis will be marked with time stamps rather than sample numbers.

    Parameters
    ----------
    state_time_course: numpy.ndarray
        The state time course to be displayed as vertical highlights.
    axis: matplotlib.axes.Axes
        Axis to plot on. Default is to create a new figure.
    colormap: str
        A matplotlib colormap from which to draw the highlight colors.
    n_time_points: int
        Number of time points to plot.
    sample_frequency: float
        The sampling frequency of the data. Default is to use sample number (i.e. 1Hz)
    highlight_kwargs: dict
        Keyword arguments to be passed to matplotlib.pyplot.axvspan.
    legend: bool
        If True a legend is added to the plot.
    fig_kwargs: dict
        Keyword arguments for matplotlib.pyplot.subplots.
    filename: str
        A file to which to save the figure.
    """
    n_time_points = min(n_time_points or np.inf, state_time_course.shape[0])

    axis_given = axis is not None
    if not axis_given:
        fig_defaults = {
            "figsize": (20, 3),
        }
        fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
        fig, axis = plt.subplots(1, **fig_kwargs)

    highlight_defaults = {"alpha": 0.2, "lw": 0}
    highlight_kwargs = override_dict_defaults(highlight_defaults, highlight_kwargs)

    n_states = state_time_course.shape[1]
    ons, offs = state_activation(state_time_course)

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
        show_or_save(filename)


def plot_cholesky(matrix, group_color_scale: bool = True):
    """Plot a matrix from its Cholesky decomposition.

    Given a Cholesky matrix, plot M @ M^T.

    Parameters
    ----------
    matrix: np.ndarray
        The matrix to plot.
    group_color_scale: bool
        If True, the colormap will be consistent across all matrices plotted.
    """
    matrix = np.array(matrix)
    c_i = from_cholesky(matrix)
    plot_matrices(c_i, group_color_scale=group_color_scale)


def plot_matrix_max_min_mean(
    matrix, group_color_scale: bool = True, cholesky: bool = False
):
    """Plot the elementwise minima, maxima and means of a matrix.

    Parameters
    ----------
    matrix: np.ndarray
        The matrix to plot.
    group_color_scale: bool
        If True, all matrices will have the same colormap scale.
    cholesky: bool
        If True, the matrix will be treated as a Cholesky decomposition.
    """
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


# noinspection PyUnresolvedReferences
def plot_matrices(
    matrix: Iterable[np.ndarray],
    group_color_scale: bool = True,
    titles: list = None,
    cmap="viridis",
    nan_color="white",
    filename: str = None,
):
    """Plot a collection of matrices.

    Given an iterable of matrices, plot each matrix in its own axis.

    Parameters
    ----------
    matrix: list of np.ndarrays
        The matrices to plot.
    group_color_scale: bool
        If True, all matrices will have the same colormap scale.
    titles: list of str
        Titles to give to each matrix axis.
    cmap: str
        Matplotlib colormap.
    nan_color: str
        Matplotlib color to use for NaN values. Default is white.
    filename: str
        A file to which to save the figure.
    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, :]
    if matrix.ndim != 3:
        raise ValueError("Must be a 3D array.")
    short, long, empty = rough_square_axes(len(matrix))
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
            v_min = matrix.min()
            v_max = matrix.max()
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

    show_or_save(filename)


def rough_square_axes(n_plots):
    """Get the most square axis layout for n_plots.

    Given n_plots, find the side lengths of the rectangle which gives the closest
    layout to a square grid of axes.

    Parameters
    ----------
    n_plots: int
        Number of plots to arrange.

    Returns
    -------
    short: int
        Number of axes on the short side.
    long: int
        Number of axes on the long side.
    empty: int
        Number of axes left blank from the rectangle.
    """
    long = np.floor(n_plots ** 0.5).astype(int)
    short = np.ceil(n_plots ** 0.5).astype(int)
    if short * long < n_plots:
        short += 1
    empty = short * long - n_plots
    return short, long, empty


def add_axis_colorbar(axis: plt.Axes):
    """Add a colorbar to an axis by compressing the axis.

    Parameters
    ----------
    axis: matplotlib.axes.Axes
    """
    try:
        pl = axis.get_images()[0]
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pl, cax=cax)
    except IndexError:
        _logger.warning("No mappable image found on axis.")


def add_figure_colorbar(fig: plt.Figure, mappable):
    fig.subplots_adjust(right=0.94)
    color_bar_axis = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    fig.colorbar(mappable, cax=color_bar_axis)


@transpose(0, "state_time_course")
def plot_state_lifetimes(
    state_time_course: np.ndarray,
    bins: int = 20,
    density: bool = False,
    match_scale_x=True,
    match_scale_y=True,
    hist_kwargs: dict = None,
    fig_kwargs: dict = None,
    filename: str = None,
):
    """Create a histogram of state lifetimes.

    For a state time course, create a histogram for each state with the distribution
    of the lengths of time for which it is active.

    Parameters
    ----------
    state_time_course: numpy.ndarray
        State time course to analyse.
    bins: int
        Number of bins for the histograms.
    density: bool
        If True, plot the probability density of the state activation lengths.
        If False, raw number.
    match_scale_x: bool
        If True, all histograms will share the same x-axis scale.
    match_scale_y: bool
        If True, all histograms will share the same y-axis scale.
    hist_kwargs: dict
        Keyword arguments to pass to matplotlib.pyplot.hist.
    fig_kwargs: dict
        Keyword arguments to pass to matplotlib.pyplot.subplots.
    filename: str
        A file to which to save the figure.
    """
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

    show_or_save(filename)


def compare_state_data(
    *state_time_courses: List[np.ndarray],
    n_time_points=None,
    sample_frequency: float = 1,
    titles: list = None,
    filename: str = None,
):
    """Plot multiple states time courses.

    For a list of state time courses, plot the state diagrams produced by
    vrad.plotting.highlight_states.

    Parameters
    ----------
    state_time_courses: list of numpy.ndarray
        List of state time courses to plot.
    n_time_points: int
        Number of time courses to plot.
    sample_frequency: float
        If given the y-axis will contain timestamps rather than sample numbers.
    titles: list of str
        Titles to give to each axis.
    """
    n_time_points = min(
        n_time_points or np.inf, *[len(stc) for stc in state_time_courses]
    )

    fig, axes = plt.subplots(
        nrows=len(state_time_courses),
        figsize=(20, 2.5 * len(state_time_courses)),
        sharex="all",
    )

    if titles is None:
        titles = [""] * len(state_time_courses)

    for state_time_course, axis, title in zip(state_time_courses, axes, titles):
        state_barcode(
            state_time_course,
            axis=axis,
            n_time_points=n_time_points,
            legend=False,
            sample_frequency=sample_frequency,
        )
        axis.set_title(title)

    add_figure_colorbar(fig=fig, mappable=fig.axes[0].get_images()[0])

    show_or_save(filename)


@transpose("state_time_course_1", 0, "state_time_course_2", 1)
def confusion_matrix(state_time_course_1: np.ndarray, state_time_course_2: np.ndarray):
    """Plot a confusion matrix.

    For two state time courses, plot the confusion matrix between each pair of states.
    The confusion matrix with the diagonal removed will also be plotted.


    Parameters
    ----------
    state_time_course_1: numpy.ndarray
        The first state time course.
    state_time_course_2: numpy.ndarray
        The second state time course.
    """
    confusion = vrad.inference.metrics.confusion_matrix(
        state_time_course_1, state_time_course_2
    )
    nan_diagonal = confusion.copy().astype(float)
    nan_diagonal[np.diag_indices_from(nan_diagonal)] = np.nan
    plot_matrices([confusion, nan_diagonal], group_color_scale=False)


def show_or_save(filename: str = None):
    if filename is None:
        plt.show()
    else:
        print(f"Saving {filename}")
        plt.savefig(filename, dpi=350)
        plt.close("all")
