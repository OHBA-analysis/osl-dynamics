"""Helper functions for plotting using matplotlib

"""
import logging
from itertools import zip_longest
from typing import Any, Iterable, List, Tuple, Union

import matplotlib
import matplotlib.patches as patches
import numpy as np
import vrad.inference.metrics
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vrad.array_ops import from_cholesky, get_one_hot, mean_diagonal
from vrad.data.task import epoch_mean
from vrad.inference.states import correlate_states, match_states, lifetimes
from vrad.utils.decorators import transpose
from vrad.utils.misc import override_dict_defaults
from vrad.utils.topoplots import Topology

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
    state_time_course: np.ndarray,
    color: str = "tab:gray",
    filename: str = None,
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
    n_states: int, colormap: str = "magma"
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


@transpose(0, 1, "time_series", "state_time_course")
def plot_state_highlighted_data(
    time_series: Union[np.ndarray, Any],
    state_time_course: np.ndarray,
    events: np.ndarray = None,
    n_samples: int = None,
    colormap: str = "magma",
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
    n_samples: int
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

    n_samples = min(
        n_samples or np.inf, state_time_course.shape[0], time_series.shape[0]
    )

    fig, axes = plt.subplots(1 if events is None else 2, **fig_kwargs, squeeze=False)

    axes = axes.ravel()

    plot_time_series(
        time_series, axes[-1], n_samples=n_samples, plot_kwargs=plot_kwargs
    )

    if events is not None:
        axes[0].plot(events[:n_samples], **event_kwargs)

    for axis in axes:
        axis.autoscale(tight=True)
        axis.axis("off")

    state_barcode(
        state_time_course,
        axes[-1],
        n_samples=n_samples,
        colormap=colormap,
        highlight_kwargs=highlight_kwargs,
        extent=[*axes[-1].get_xlim(), *axes[-1].get_ylim()],
    )

    plt.tight_layout()

    show_or_save(filename)


@transpose("time_series", 0)
def plot_time_series(
    time_series: np.ndarray,
    axis: plt.Axes = None,
    n_samples: int = None,
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
    n_samples: int
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
    time_series = np.asarray(time_series)
    n_samples = min(n_samples or np.inf, time_series.shape[0])
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
        np.maximum(time_series[:n_samples].max(), time_series[:n_samples].min()) * 1.2
    )
    gaps = np.arange(n_channels)[::-1] * separation

    axis.plot(time_series[:n_samples] + gaps[None, :], **plot_kwargs)

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
    colormap: str = "magma",
    n_samples: int = None,
    sampling_frequency: float = 1,
    highlight_kwargs: dict = None,
    legend: bool = True,
    fig_kwargs: dict = None,
    filename: str = None,
    extent: List[float] = None,
):
    """Create a barcode plot for a state time course.

    Given a state time course either expressed as an 1D array of integer state
    activations or as a 2D array of one-hot encoded state activations, produce a
    barcode plot with each state represented by a color drawn from the colormap
    provided.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course as a 1D or 2D array.
    axis : matplotlib.pyplot.Axes
        Optional axis on which to plot the barcode.
    colormap : str
        Matplotlib colormap name.
    n_samples : int
        Number of samples to plot.
    sampling_frequency : str
        Sampling frequency of the data to allow for x-axis to be time
         rather than sample number.
    highlight_kwargs : dict
        Keyword arguments to pass to imshow.
    legend : bool
        Toggle whether a legend is displayed.
    fig_kwargs : dict
        Keyword arguments to pass to plt.subplots.
    filename : str
        Name of file to save plot to.
    extent : list of float
        Extent to pass to imshow.
    """
    state_time_course = np.asarray(state_time_course)
    if state_time_course.ndim == 2:
        state_time_course = state_time_course.argmax(axis=1)
    if state_time_course.ndim != 1:
        raise ValueError("state_time_course must be 1D or 2D.")

    n_samples = min(n_samples or np.inf, len(state_time_course))
    state_time_course = state_time_course[:n_samples]

    axis_given = axis is not None
    if not axis_given:
        fig_defaults = {
            "figsize": (24, 2.5),
        }
        fig_kwargs = override_dict_defaults(fig_defaults, fig_kwargs)
        fig, axis = plt.subplots(1, **fig_kwargs)

    highlight_defaults = {}
    highlight_kwargs = override_dict_defaults(highlight_defaults, highlight_kwargs)

    n_states = np.max(np.unique(state_time_course)) + 1

    cmap = plt.cm.get_cmap(colormap, lut=n_states)

    extent = extent or [0, n_samples / sampling_frequency, 0, 1]

    axis.imshow(
        state_time_course[None],
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=np.max(n_states) - 0.5,
        interpolation="none",
        extent=extent,
        **highlight_kwargs,
    )

    plt.setp(axis.spines.values(), visible=False)

    if legend:
        add_axis_colorbar(axis)

    axis.set_yticks([])

    if not axis_given:
        show_or_save(filename)


def plot_covariance_from_cholesky(matrix, group_color_scale: bool = True):
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
    main_title: str = None,
    cmap: str = "viridis",
    nan_color: str = "white",
    log_norm: bool = False,
    filename: str = None,
):
    """Plot a collection of matrices.

    Given an iterable of matrices, plot each matrix in its own axis. The axes are
    arranged as close to a square (N x N axis grid) as possible.

    Parameters
    ----------
    matrix: list of np.ndarrays
        The matrices to plot.
    group_color_scale: bool
        If True, all matrices will have the same colormap scale, where we use the
        minimum and maximum across all matrices as the scale.
    titles: list of str
        Titles to give to each matrix axis.
    main_title: str
        Main title to be placed at the top of the plot.
    cmap: str
        Matplotlib colormap.
    nan_color: str
        Matplotlib color to use for NaN values. Default is white.
    log_norm: bool
        Should we show the elements on a log scale? Default is False.
    filename: str
        A file to which to save the figure.
    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, :]
    if matrix.ndim != 3:
        raise ValueError("Must be a 3D array.")
    short, long, empty = rough_square_axes(len(matrix))
    fig, axes = plt.subplots(ncols=short, nrows=long, squeeze=False)

    if titles is None:
        titles = [""] * len(matrix)

    cmap = matplotlib.cm.get_cmap(cmap).copy()
    cmap.set_bad(color=nan_color)

    for grid, axis, title in zip_longest(matrix, axes.ravel(), titles):
        if grid is None:
            axis.remove()
            continue
        if group_color_scale:
            v_min = matrix.min()
            v_max = matrix.max()
            if log_norm:
                im = axis.matshow(grid, cmap=cmap, norm=LogNorm(vmin=v_min, vmax=v_max))
            else:
                im = axis.matshow(grid, vmin=v_min, vmax=v_max, cmap=cmap)
        else:
            if log_norm:
                im = axis.matshow(grid, cmap=cmap, norm=LogNorm())
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

    fig.suptitle(main_title)

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
        return plt.colorbar(pl, cax=cax)
    except IndexError:
        _logger.warning("No mappable image found on axis.")


def add_figure_colorbar(fig: plt.Figure, mappable):
    """Add a colorbar to a figure based on a mappable object.

    Adds an extra axis to a figure to add a colorbar. The colors are allocated based on
    a mappable object provided.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
    mappable
        An object from which matplotlib can create a colorbar.
    """
    fig.subplots_adjust(right=0.94)
    color_bar_axis = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    return fig.colorbar(mappable, cax=color_bar_axis)


@transpose(0, "state_time_course")
def plot_state_lifetimes(
    state_time_course: np.ndarray,
    bins: int = "auto",
    density: bool = False,
    match_scale_x: bool = False,
    match_scale_y: bool = False,
    x_range: list = None,
    x_label: str = None,
    y_label: str = None,
    hist_kwargs: dict = None,
    fig_kwargs: dict = None,
    filename: str = None,
):
    """Create a histogram of state lifetimes.

    For a state time course, create a histogram for each state with the distribution
    of the lengths of time for which it is active.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course to analyse.
    bins : int
        Number of bins for the histograms.
    density : bool
        If True, plot the probability density of the state activation lengths.
        If False, raw number.
    match_scale_x : bool
        If True, all histograms will share the same x-axis scale.
    match_scale_y : bool
        If True, all histograms will share the same y-axis scale.
    x_range : list
        The limits on the values presented on the x-axis.
    x_label : str
        x-axis label.
    y_label : str
        y-axis label.
    hist_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.hist.
    fig_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.subplots.
    filename : str
        A file to which to save the figure.
    """
    if state_time_course.ndim == 1:
        state_time_course = get_one_hot(state_time_course)
    if state_time_course.ndim != 2:
        raise ValueError("state_timecourse must be a 2D array")

    channel_lifetimes = lifetimes(state_time_course)
    n_plots = state_time_course.shape[1]
    short, long, empty = rough_square_axes(n_plots)

    colors = get_colors(n_plots)

    default_hist_kwargs = {}
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
        if x_range is not None:
            if len(x_range) != 2:
                raise ValueError("x_range must be [x_min, x_max].")
            axis.set_xlim(x_range[0], x_range[1])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

    plt.tight_layout()
    show_or_save(filename)


def plot_separate_time_series(
    *time_series,
    n_samples: int = None,
    sampling_frequency: float = None,
    plot_kwargs: dict = None,
    fig_kwargs: dict = None,
    filename: str = None,
):
    """Plot time series as separate subplots.

    Parameters
    ----------
    time_series : list of numpy.ndarray
        Time series to be plotted. Should be (n_samples, n_lines).
    sampling_frequency: float
        Sampling frequency of the input data, enabling us to label the x-axis(!)
    n_samples : int
        Number of samples to be shown on the x-axis.
    plot_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.plot.
    fig_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.subplots.
    filename : str
        Filename to save figure to.
    """

    time_series = np.asarray(time_series)
    n_samples = n_samples or min([stc.shape[0] for stc in time_series])
    n_lines = time_series[0].shape[1]

    if sampling_frequency is not None:
        time_vector = np.linspace(0, n_samples / sampling_frequency, n_samples)
    else:
        time_vector = np.linspace(0, n_samples, n_samples)

    default_fig_kwargs = {"figsize": (20, 10), "sharex": "all"}
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)
    fig, axes = plt.subplots(n_lines, **fig_kwargs)

    default_plot_kwargs = {"lw": 0.7}
    plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    for group in time_series:
        for axis, line in zip(axes, group.T):
            axis.plot(time_vector, line[:n_samples], **plot_kwargs)
            axis.autoscale(axis="x", tight=True)

    if sampling_frequency is not None:
        axes[-1].set_xlabel("Time (s)")
    else:
        axes[-1].set_xlabel("Samples")

    plt.tight_layout()
    show_or_save(filename)


def compare_state_data(
    *state_time_courses: List[np.ndarray],
    n_samples: int = None,
    sampling_frequency: float = 1.0,
    titles: list = None,
    x_label: str = None,
    filename: str = None,
    **kwargs: dict,
):
    """Plot multiple states time courses.

    For a list of state time courses, plot the state diagrams produced by
    vrad.utils.plotting.state_barcode.

    Parameters
    ----------
    state_time_courses: list of numpy.ndarray
        List of state time courses to plot.
    n_samples: int
        Number of time courses to plot.
    sampling_frequency: float
        If given the y-axis will contain timestamps rather than sample numbers.
    titles: list of str
        Titles to give to each axis.
    x_label : str
        x-axis label.
    filename: str
        Filename to save figure to.
    kwargs: dict
        Keyword arguments passed to state_barcode.
    """
    n_samples = min(n_samples or np.inf, *[len(stc) for stc in state_time_courses])

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
            n_samples=n_samples,
            legend=False,
            sampling_frequency=sampling_frequency,
            **kwargs,
        )
        axis.set_title(title)

    axes[-1].set_xlabel(x_label)
    plt.tight_layout()
    add_figure_colorbar(fig=fig, mappable=fig.axes[0].get_images()[0])
    show_or_save(filename)


@transpose("state_time_course_1", 0, "state_time_course_2", 1)
def plot_confusion_matrix(
    state_time_course_1: np.ndarray,
    state_time_course_2: np.ndarray,
    filename: str = None,
):
    """Plot a confusion matrix.

    For two state time courses, plot the confusion matrix between each pair of states.
    The confusion matrix with the diagonal removed will also be plotted.


    Parameters
    ----------
    state_time_course_1: numpy.ndarray
        The first state time course.
    state_time_course_2: numpy.ndarray
        The second state time course.
    filename : str
        Output filename. Optional.
    """
    confusion = vrad.inference.metrics.confusion_matrix(
        state_time_course_1, state_time_course_2
    )
    nan_diagonal = confusion.copy().astype(float)
    nan_diagonal[np.diag_indices_from(nan_diagonal)] = np.nan
    plot_matrices([confusion, nan_diagonal], group_color_scale=False, filename=filename)


def plot_line(
    x: list,
    y: list,
    labels: list = None,
    legend_loc: int = 1,
    errors: list = None,
    x_range: list = None,
    y_range: list = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    figsize: tuple = (7, 4),
    filename: str = None,
):
    """Basic line plot.

    Parameters
    ----------
    x : list of numpy arrays
        x-ordinates.
    y : list of numpy arrays
        y-ordinates.
    labels : list of str
        Legend labels for each line.
    legend_loc : int
        Matplotlib legend location identifier. Optional. Default is top right.
    errors : list with 2 items
        Min and max errors. Optional.
    x_range : list
        Minimum and maximum for x-axis. Optional.
    y_range : list
        Minimum and maximum for y-axis. Optional.
    x_label : str
        Label for x-axis. Optional.
    y_label : str
        Label for y-axis. Optional.
    title : str
        Figure title. Optional.
    figsize : tuple
        Figure size in inches. Optional, default is (7, 4).
    filename : str
        Output filename. Optional.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if len(x) != len(y):
        raise ValueError("Different number of x and y arrays given.")

    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        else:
            if len(labels) != len(x):
                raise ValueError("Incorrect number of lines or labels passed.")
        add_legend = True
    else:
        labels = [None] * len(x)
        add_legend = False

    if errors is None:
        errors_min = [None] * len(x)
        errors_max = [None] * len(x)
    elif len(errors) != 2:
        raise ValueError(
            "Errors must be errors=[[y_min1, y_min2,...], [y_max1, y_max2,..]]."
        )
    elif len(errors[0]) != len(x) or len(errors[1]) != len(x):
        raise ValueError("Incorrect number of errors passed.")
    else:
        errors_min = errors[0]
        errors_max = errors[1]

    # Plot lines
    for (x_data, y_data, label, e_min, e_max) in zip(
        x, y, labels, errors_min, errors_max
    ):
        ax.plot(x_data, y_data, label=label)
        if e_min is not None:
            ax.fill_between(x_data, e_min, e_max, alpha=0.3)

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if add_legend:
        ax.legend(loc=legend_loc)

    # Clean up layout and show or save
    plt.tight_layout()
    show_or_save(filename)


def plot_scatter(
    x: list,
    y: list,
    labels: list = None,
    legend_loc: int = 1,
    errors: list = None,
    x_range: list = None,
    y_range: list = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    figsize: tuple = (7, 4),
    markers: list = None,
    marker_size: float = None,
    filename: str = None,
):
    """Basic scatter plot.

    Parameters
    ----------
    x : list of numpy arrays
        x-ordinates.
    y : list of numpy arrays
        y-ordinates.
    labels : list of str
        Legend labels for each line.
    legend_loc : int
        Matplotlib legend location identifier. Optional. Default is top right.
    errors : list
        Error bars. Optional.
    x_range : list
        Minimum and maximum for x-axis. Optional.
    y_range : list
        Minimum and maximum for y-axis. Optional.
    x_label : str
        Label for x-axis. Optional.
    y_label : str
        Label for y-axis. Optional.
    title : str
        Figure title. Optional.
    figsize : tuple
        Figure size in inches. Optional, default is (7, 4).
    markers : list of str
        Markers to used for each set of data points. Optional.
    marker_size : float
        Size of markers. Optional.
    filename : str
        Output filename. Optional.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if len(x) != len(y):
        raise ValueError("Different number of x and y arrays given.")

    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        else:
            if len(labels) != len(x):
                raise ValueError("Incorrect number of data points or labels passed.")
        add_legend = True
    else:
        labels = [None] * len(x)
        add_legend = False

    if errors is None:
        errors = [None] * len(x)

    if markers is not None:
        if len(markers) != len(x):
            raise ValueError("Incorrect number of data points or markers passed.")
    else:
        markers = [None] * len(x)

    # Colours
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    if len(x) > len(colors):
        raise ValueError("Too many data points passed for the color cycle.")

    # Plot data
    for i in range(len(x)):
        ax.scatter(x[i], y[i], label=labels[i], marker=markers[i], s=marker_size)
        if errors[i] is not None:
            ax.errorbar(x[i], y[i], yerr=errors[i], fmt="none", c=colors[i])

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if add_legend:
        ax.legend(loc=legend_loc)

    # Clean up layout and show or save
    plt.tight_layout()
    show_or_save(filename)


def plot_hist(
    data: list,
    bins: list,
    labels: list = None,
    legend_loc: int = 1,
    x_range: list = None,
    y_range: list = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    filename: str = None,
):
    """Basic histogram plot.

    Parameters
    ----------
    data : list of np.ndarray
        Data to plot.
    bins : list of int
        Number of bins for each item in data.
    labels : list of str
        Legend labels for each line.
    legend_loc : int
        Matplotlib legend location identifier. Optional. Default is top right.
    x_range : list
        Minimum and maximum for x-axis. Optional.
    y_range : list
        Minimum and maximum for y-axis. Optional.
    x_label : str
        Label for x-axis. Optional.
    y_label : str
        Label for y-axis. Optional.
    title : str
        Figure title. Optional.
    filename : str
        Output filename. Optional.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    if len(data) != len(bins):
        raise ValueError("Different number of bins and data.")

    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        else:
            if len(labels) != len(data):
                raise ValueError("Incorrect number of labels or data passed.")
        add_legend = True
    else:
        labels = [None] * len(data)
        add_legend = False

    # Plot histograms
    for (d, b, l) in zip(data, bins, labels):
        ax.hist(d, bins=b, label=l, histtype="step")

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if add_legend:
        ax.legend(loc=legend_loc)

    # Clean up layout and show or save
    plt.tight_layout()
    show_or_save(filename)


def show_or_save(filename: str = None):
    """Either show or save the current figure.

    Parameters
    ----------
    filename : str
    """
    if filename is None:
        plt.show()
    else:
        print(f"Saving {filename}")
        plt.savefig(filename, dpi=350)
        plt.close("all")


def topoplot(
    layout: str,
    data: np.ndarray,
    channel_names: List[str] = None,
    plot_boxes: bool = False,
    show_deleted_sensors: bool = False,
    show_names: bool = False,
    title: str = None,
    colorbar: bool = True,
    axis: plt.Axes = None,
    cmap: str = "plasma",
    n_contours: int = 10,
    filename: str = None,
):
    """Make a contour plot in sensor space.

    Create a contour plot by interpolating a field from a set of values provided for
    each sensor location in an MEG layout. Within the context of VRAD this is likely
    to be an array of (all positive) values taken from the diagonal of a covariance
    matrix, but one can also plot any sensor level M/EEG data.

    Parameters
    ----------
    layout: str
        The name of an MEG layout (matching one from FieldTrip).
    data: numpy.ndarray
        The value of the field at each sensor.
    channel_names: List[str]
        A list of channel names which are present in the
        data (removes missing channels).
    plot_boxes: bool
        Show boxes representing the height and width of sensors.
    show_deleted_sensors: bool
        Show sensors missing from `channel_names` in red.
    show_names: bool
        Show the names of channels (can get very cluttered).
    title: str
        A title for the figure.
    colorbar: bool
        Show a colorbar for the field.
    axis: matplotlib.pyplot.Axes
        matplotlib axis to plot on.
    cmap: str
        matplotlib colourmap. Defaults to 'plasma'
    n_contours: int
        number of field isolines to show on the plot. Defaults to 10.
    filename : str
        Output filename. Optional.
    """
    topology = Topology(layout)

    if channel_names is not None:
        topology.keep_channels(channel_names)

    fig = topology.plot_data(
        data,
        plot_boxes=plot_boxes,
        show_deleted_sensors=show_deleted_sensors,
        show_names=show_names,
        title=title,
        colorbar=colorbar,
        axis=axis,
        cmap=cmap,
        n_contours=n_contours,
    )

    if filename is not None:
        print(f"Saving {filename}")
        plt.savefig(filename, dpi=350)

    return fig


def plot_connections(
    weights: np.ndarray,
    labels: List[str] = None,
    ax: plt.Axes = None,
    cmap: str = "hot",
    text_color: str = None,
) -> plt.Axes:
    """Create a chord diagram representing the values of a matrix.

    For a matrix of weights, create a chord diagram where the color of the line
    connecting two nodes represents the value indexed by the position of the nodes in
    the lower triangle of the matrix.

    This is useful for showing things like co-activation between sensors/parcels or
    relations between nodes in a network.

    Parameters
    ----------
    weights: numpy.ndarray
        An NxN matrix of weights.
    labels: list of str
        A name for each node in the weights matrix (e.g. parcel names)
    ax: matplotlib.pyplot.Axes
        A matplotlib axis on which to plot.
    cmap: str
        A string corresponding to a matplotlib colormap.
    text_color: str
        A string corresponding to a matplotlib color.

    Returns
    -------
    ax: matplotlib.pyplot.Axes

    Examples
    --------
    >>> rng = np.random.default_rng(seed=42)
    >>> covariance = rng.normal(size=(38, 38))
    >>> covariance[[24, 25, 26, 27], [12,13,14,15]] = 4
    >>> plot_connections(
    ...     covariance,
    ...     (f"ROI {i + 1}" for i in range(covariance.shape[0])),
    ...     cmap="magma",
    ...     text_color="white"
    ... )
    >>> plt.show()
    """
    weights = np.abs(weights)
    x, y = np.diag_indices_from(weights)
    weights[x, y] = 0
    weights /= weights.max()

    inner = 0.9
    outer = 1.0

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    highest_color = cmap(norm(1)) if text_color is None else text_color
    zero_color = cmap(norm(0))

    text_color = {
        "text.color": highest_color,
        "axes.labelcolor": highest_color,
        "xtick.color": highest_color,
        "ytick.color": highest_color,
    }

    angle = np.radians(360 / weights.shape[0])
    pad = np.radians(0.5)

    starts = np.arange(0, 2 * np.pi, angle)
    lefts = starts + pad
    rights = starts + angle - pad
    centers = 0.5 * (lefts + rights)

    with matplotlib.rc_context(text_color):
        if not ax:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

        for left, right in zip(lefts, rights):
            verts = [
                (left, inner),
                (left, outer),
                (right, outer),
                (right, inner),
                (0.0, 0.0),
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            path = Path(verts, codes)

            patch = patches.PathPatch(path, facecolor="orange", lw=1)
            ax.add_patch(patch)
        ax.set_yticks([])
        ax.grid(False)

        bezier_codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        rebound = 0.5

        for i, j in zip(*np.tril_indices_from(weights)):
            center_1 = centers[i]
            center_2 = centers[j]

            verts = [
                (center_1, inner),
                (center_1, rebound),
                (center_2, rebound),
                (center_2, inner),
            ]

            path = Path(verts, bezier_codes)

            patch = patches.PathPatch(
                path,
                facecolor="none",
                lw=2,
                edgecolor=cmap(weights[i, j]),
                alpha=weights[i, j] ** 2,
            )
            ax.add_patch(patch)

        ax.set_xticks([])

        ax.set_facecolor(zero_color)
        fig.patch.set_facecolor(zero_color)

        if labels is None:
            labels = [""] * len(centers)
        for center, label in zip(centers, labels):
            rotation = np.degrees(center)

            if 0 <= rotation < 90:
                horizontal_alignment = "left"
                vertical_alignment = "bottom"
            elif 90 <= rotation < 180:
                horizontal_alignment = "right"
                vertical_alignment = "bottom"
            elif 180 <= rotation < 270:
                horizontal_alignment = "right"
                vertical_alignment = "top"
            else:
                horizontal_alignment = "left"
                vertical_alignment = "top"

            if 90 <= rotation < 270:
                rotation += 180

            ax.annotate(
                label,
                (center, outer + 0.05),
                rotation=rotation,
                horizontalalignment=horizontal_alignment,
                verticalalignment=vertical_alignment,
            )

        ax.autoscale_view()
        plt.setp(ax.spines.values(), visible=False)

    return ax


def state_stackplots(
    *state_time_courses: np.ndarray,
    n_samples: int,
    cmap: str = "viridis",
    sampling_frequency: float = None,
    fig_kwargs: dict = None,
    filename: str = None,
    y_label: str = "State Activation",
    title: str = None,
    **stackplot_kwargs: dict,
) -> plt.Figure:
    """

    Parameters
    ----------
    state_time_courses : numpy.ndarray
        A selection of state time courses passed as separate arguments.
    n_samples: int
        Number of time points to be plotted.
    cmap : str
        A matplotlib colormap string.
    sampling_frequency : float
        The sampling frequency of the data in Hertz.
    fig_kwargs : dict
        Any parameters to be passed to the matplotlib.pyplot.subplots constructor.
    y_label : str
        Label for y-axis.
    title : str
        Title for the plot.
    stackplot_kwargs : dict
        Any parameters to be passed to matplotlib.pyplot.stackplot.

    Returns
    -------
    plt.Figure
    """
    n_states = max(stc.shape[1] for stc in state_time_courses)
    n_samples = min(n_samples or np.inf, state_time_courses[0].shape[0])
    cmap = plt.cm.get_cmap(name=cmap, lut=n_states)
    colors = cmap.colors

    n_stcs = len(state_time_courses)

    default_fig_kwargs = dict(
        figsize=(12, 2.5 * n_stcs), sharex="all", facecolor="white"
    )
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    default_stackplot_kwargs = dict(colors=colors)
    stackplot_kwargs = override_dict_defaults(
        default_stackplot_kwargs, stackplot_kwargs
    )

    fig, axes = plt.subplots(nrows=n_stcs, **fig_kwargs)
    if isinstance(axes, plt.Axes):
        axes = [axes]

    for stc, axis in zip(state_time_courses, axes):
        time_vector = (
            np.arange(n_samples) / sampling_frequency
            if sampling_frequency
            else range(n_samples)
        )
        axis.stackplot(time_vector, stc[:n_samples].T, **stackplot_kwargs)
        axis.autoscale(tight=True)
        axis.set_ylabel(y_label)

    axes[-1].set_xlabel("Time (s)" if sampling_frequency else "Sample")
    axes[0].set_title(title)

    fig.tight_layout()

    add_figure_colorbar(
        fig,
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=n_states - 0.5), cmap=cmap
        ),
    )

    show_or_save(filename)


def plot_epoched(
    data: np.ndarray,
    time_index: np.ndarray,
    sampling_frequency: float = None,
    pre: int = 125,
    post: int = 1000,
    title: str = None,
    legend: bool = True,
) -> plt.Figure:
    """Plot continuous data, epoched and meaned over epochs.

    Parameters
    ----------

    data: numpy.ndarray
        A [time x channels] dataset to be epoched.
    time_index: numpy.ndarray
        The integer indices of the start of each epoch.
    sampling_frequency
        The sampling frequency of the data in Hertz.
    pre: int
        The integer number of samples to include before the trigger.
    post: int
        The integer number of samples to include after the trigger.
    title: str
        Title of the figure.
    legend: bool
        Should a legend be created (default: True).

    Returns
    -------
    epoch_mean_figure: matplotlib.pyplot.Figure
        A figure of data meaned over epochs.

    """
    epoched_1 = epoch_mean(data, time_index, pre, post)

    x_label = "Samples"
    time_index = np.arange(-pre, post)
    if sampling_frequency:
        time_index = time_index / sampling_frequency
        x_label = "Time (s)"

    fig, axis = plt.subplots(1, figsize=(20, 3))
    for i, s in enumerate(epoched_1.T):
        axis.plot(time_index, s, label=i)
    axis.axvline(0, c="k")
    axis.autoscale(axis="x", tight=True)
    if title:
        axis.set_title(title)
    if legend:
        axis.legend(loc="upper right")
    axis.set_xlabel(x_label)
    return fig
