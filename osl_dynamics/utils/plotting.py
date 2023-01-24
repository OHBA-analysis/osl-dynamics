"""Plotting functions.

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from itertools import zip_longest

from osl_dynamics.array_ops import get_one_hot
from osl_dynamics.utils.misc import override_dict_defaults
from osl_dynamics.utils.topoplots import Topology


QUAL_CMAPS = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]


def set_style(params):
    """Sets matplotlib's style.

    Wrapper for plt.rcParams.update(). List of parameters can be found here:
    https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams

    Parameters
    ----------
    params : dict
        Dictionary of style parameters to update.
    """
    plt.rcParams.update(params)


def create_figure(*args, **kwargs):
    """Creates matplotlib figure and axes objects.

    Wrapper for plt.subplots().

    Parameters
    ----------
    fig_kwargs
        Arguments to pass to plt.subplots().
    """
    fig, ax = plt.subplots(*args, **kwargs)
    return fig, ax


def show(tight_layout=True):
    """Displays all figures in memory.

    Wrapper for plt.show().

    Parameters
    ----------
    tight_layout : bool
        Should we call plt.tight_layout()?
    """
    if tight_layout:
        plt.tight_layout()
    plt.show()


def save(fig, filename, tight_layout=True):
    """Saves a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        matplotlib figure object.
    filename : str
        Output filename.
    tight_layout : bool
        Should we call fig.tight_layout()?
    """
    print(f"Saving {filename}")
    if tight_layout:
        fig.tight_layout()
    fig.savefig(filename)


def close(fig=None):
    """Close a figure.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        Figure to close. Defaults to all figures.
    """
    if fig is None:
        fig = "all"
    plt.close(fig)


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
    long = np.floor(n_plots**0.5).astype(int)
    short = np.ceil(n_plots**0.5).astype(int)
    if short * long < n_plots:
        short += 1
    empty = short * long - n_plots
    return short, long, empty


def get_colors(n, colormap="magma"):
    """Produce equidistant colors from a matplotlib colormap.

    Given a matplotlib colormap, produce a series of RGBA colors which are equally
    spaced by value. There is no guarantee that these colors will be perceptually
    uniformly distributed and with many colors will likely be extremely close.
    Alpha is 1.0 for all colors.

    Parameters
    ----------
    n : int
        The number of colors to return.
    colormap : str
        The name of a matplotlib colormap.

    Returns
    -------
    colors: list of tuple of float
        A list of colors in RGBA format. A = 1.0 in all cases.
    """
    colormap = plt.get_cmap(colormap)
    colors = [colormap(1 * i / n) for i in range(n)]
    return colors


def plot_line(
    x,
    y,
    labels=None,
    legend_loc=1,
    errors=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
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
        Matplotlib legend location identifier. Default is top right.
    errors : list with 2 items
        Min and max errors.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    plot_kwargs : dict
        Arguments to pass to the ax.plot method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
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

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot lines
    for (x_data, y_data, label, e_min, e_max) in zip(
        x, y, labels, errors_min, errors_max
    ):
        ax.plot(x_data, y_data, label=label, **plot_kwargs)
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

    # Save figure
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_scatter(
    x,
    y,
    labels=None,
    legend_loc=1,
    errors=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    markers=None,
    annotate=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
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
        Matplotlib legend location identifier. Default is top right.
    errors : list
        Error bars.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    markers : list of str
        Markers to used for each set of data points.
    annotate : List of array like objects
        Annotation for each data point for each set of data points.
    plot_kwargs : dict
        Arguments to pass to the ax.scatter method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
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

    if annotate is not None:
        if len(annotate) != len(x):
            raise ValueError("Incorrect number of data points or annotates passed.")
    else:
        annotate = [None] * len(x)

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    if len(x) > 10:
        colors = get_colors(len(x), colormap="tab20")
    else:
        colors = get_colors(len(x), colormap="tab10")

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot data
    for i in range(len(x)):
        ax.scatter(
            x[i],
            y[i],
            label=labels[i],
            marker=markers[i],
            color=colors[i],
            **plot_kwargs,
        )
        if errors[i] is not None:
            ax.errorbar(x[i], y[i], yerr=errors[i], fmt="none", c=colors[i])
        if annotate[i] is not None:
            for j, txt in enumerate(annotate[i]):
                ax.annotate(txt, (x[i][j], y[i][j]))

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

    # Save figure
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_hist(
    data,
    bins,
    labels=None,
    legend_loc=1,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Basic histogram plot.

    Parameters
    ----------
    data : list of np.ndarray
        Raw data to plot (i.e. non-histogramed data).
    bins : list of int
        Number of bins for each item in data.
    labels : list of str
        Legend labels for each line.
    legend_loc : int
        Matplotlib legend location identifier. Default is top right.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    plot_kwargs : dict
        Arguments to pass to the ax.hist method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
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

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

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

    # Save the figure if a filename has been pass
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_bar_chart(
    counts,
    x=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Bar chart plot.

    Parameters
    ----------
    counts : list of np.ndarray
        Data to plot.
    x : list or np.ndarray
        x-values for counts.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    plot_kwargs : dict
        Arguments to pass to the ax.bar method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
    if x is None:
        x = range(1, len(counts) + 1)
    elif len(x) != len(counts):
        raise ValueError("Incorrect number of x-values or counts passed.")
    else:
        x = [str(xi) for xi in x]

    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot bar chart
    ax.bar(x, counts, **plot_kwargs)

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Save the figure if a filename has been pass
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_gmm(
    data,
    amplitudes,
    means,
    stddevs,
    bins=50,
    legend_loc=1,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot a two component Gaussian mixture model.

    Parameters
    ----------
    data : np.ndarray
        Raw data to plot as a histogram.
    means : np.ndarray
        Mean of each Gaussian component.
    stddevs : np.ndarray
        Standard deviation of each Gaussian component.
    bins : list of int
        Number of bins for the historgram.
    legend_loc : int
        Position for the legend.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    plot_kwargs : dict
        Arguments to pass to the ax.hist method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot histogram
    ax.hist(data, bins=bins, histtype="step", density=True)

    # Plot Gaussian components
    x = np.arange(min(data), max(data), (max(data) - min(data)) / bins)
    y1 = amplitudes[0] * np.exp(-((x - means[0]) ** 2) / (2 * stddevs[0] ** 2))
    y2 = amplitudes[1] * np.exp(-((x - means[1]) ** 2) / (2 * stddevs[1] ** 2))
    ax.plot(x, y1, label="Off")
    ax.plot(x, y2, label="On")
    ax.plot(x, y1 + y2)

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add legend
    ax.legend(loc=legend_loc)

    # Save the figure if a filename has been pass
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_violin(
    data,
    show_mean=True,
    show_median=True,
    legend_loc=1,
    x=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Violin plot.

    Parameters
    ----------
    data : list of np.ndarray
        Data to plot.
    show_mean : bool
        Should we show the mean?
    show_median : bool
        Should we show the median?
    legend_loc : int
        Position for the legend.
    x : list or np.ndarray
        x-values for data.
    x_range : list
        Minimum and maximum for x-axis.
    y_range : list
        Minimum and maximum for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str
        Figure title.
    plot_kwargs : dict
        Arguments to pass to the ax.violinplot method.
    fig_kwargs : dict
        Arguments to pass to plt.subplots.
    ax : matplotlib.axes.axes
        Axis object to plot on.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """

    # Validation
    if x is None:
        x = range(len(data))
    elif len(x) != len(data):
        raise ValueError("Incorrect number of x-values or data passed.")
    else:
        x = [str(xi) for xi in x]

    if x_range is None:
        x_range = [None, None]

    if y_range is None:
        y_range = [None, None]

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (7, 4)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Replace emtpy lists in data with a pair of nans
    data = [np.array([np.nan, np.nan]) if len(d) == 0 else d for d in data]

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot violins
    ax.violinplot(data, positions=range(len(x)), showextrema=False)
    if show_mean:
        ax.scatter(
            x, [np.mean(d) for d in data], label="Mean", marker="+", color="black"
        )
    if show_median:
        ax.scatter(
            x, [np.median(d) for d in data], label="Median", marker="x", color="black"
        )

    # Set axis range
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if show_mean and show_median:
        ax.legend(loc=legend_loc)

    # Save the figure if a filename has been pass
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_time_series(
    time_series,
    n_samples=None,
    y_tick_values=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot a time series with channel separation.

    Parameters
    ----------
    time_series : numpy.ndarray
        The time series to be plotted. Shape must be (n_samples, n_channels).
    n_samples : int
        The number of time points to be plotted.
    y_tick_values:
        Labels for the channels to be placed on the y-axis.
    fig_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.subplots.
    plot_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.plot.
    ax : matplotlib.axes.Axes
        The axis on which to plot the data. If not given, a new axis is created.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    time_series = np.asarray(time_series)
    n_samples = min(n_samples or np.inf, time_series.shape[0])
    n_channels = time_series.shape[1]

    # Validation
    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (12, 8)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    default_plot_kwargs = {"lw": 0.7, "color": "tab:blue"}
    if plot_kwargs is None:
        plot_kwargs = default_plot_kwargs
    else:
        plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    # Calculate separation
    separation = (
        np.maximum(time_series[:n_samples].max(), time_series[:n_samples].min()) * 1.2
    )
    gaps = np.arange(n_channels)[::-1] * separation

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Plot data
    ax.plot(time_series[:n_samples] + gaps[None, :], **plot_kwargs)

    ax.autoscale(tight=True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set x and y axis tick labels
    ax.set_xticks([])
    if y_tick_values is not None:
        ax.set_yticks(gaps)
        ax.set_yticklabels(y_tick_values)
    else:
        ax.set_yticks([])

    # Save figure
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_separate_time_series(
    *time_series,
    n_samples=None,
    sampling_frequency=None,
    fig_kwargs=None,
    plot_kwargs=None,
    filename=None,
):
    """Plot time series as separate subplots.

    Parameters
    ----------
    time_series : numpy.ndarrays
        Time series to be plotted. Should be (n_samples, n_lines). Each line
        is its own subplot.
    sampling_frequency: float
        Sampling frequency of the input data, enabling us to label the x-axis.
    n_samples : int
        Number of samples to be shown on the x-axis.
    fig_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.subplots.
    plot_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.plot.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    time_series = np.asarray(time_series)
    n_samples = n_samples or min([ts.shape[0] for ts in time_series])
    n_lines = time_series[0].shape[1]

    default_fig_kwargs = {"figsize": (20, 10), "sharex": "all"}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    default_plot_kwargs = {"lw": 0.7}
    if plot_kwargs is None:
        plot_kwargs = default_plot_kwargs
    else:
        plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    if sampling_frequency is not None:
        time_vector = np.linspace(0, n_samples / sampling_frequency, n_samples)
    else:
        time_vector = np.linspace(0, n_samples, n_samples)

    # Create figure
    fig, axes = create_figure(n_lines, **fig_kwargs)
    if n_lines == 1:
        axes = [axes]

    # Plot each time series
    for group in time_series:
        for axis, line in zip(axes, group.T):
            axis.plot(time_vector, line[:n_samples], **plot_kwargs)
            axis.autoscale(axis="x", tight=True)

    # Label the x-axis
    if sampling_frequency is not None:
        axes[-1].set_xlabel("Time (s)")
    else:
        axes[-1].set_xlabel("Sample")

    # Save figure
    if filename is not None:
        save(fig, filename)

    return fig, axes


def plot_epoched_time_series(
    data,
    time_index,
    sampling_frequency=None,
    pre=125,
    post=1000,
    baseline_correct=False,
    legend=True,
    legend_loc=1,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot continuous data, epoched and meaned over epochs.

    Parameters
    ----------
    data : numpy.ndarray
        A [time x channels] dataset to be epoched.
    time_index : numpy.ndarray
        The integer indices of the start of each epoch.
    sampling_frequency : float
        The sampling frequency of the data in Hertz.
    pre : int
        The integer number of samples to include before the trigger.
    post : int
        The integer number of samples to include after the trigger.
    baseline_correct : bool
        Should we subtract the mean value pre-trigger.
    legend : bool
        Should a legend be created.
    legend_loc : int
        Location of the legend.
    title : str
        Title of the figure.
    fig_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.subplots.
    plot_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.plot.
    ax : matplotlib.axes.Axes
        The axis on which to plot the data. If not given, a new axis is created.
    filename : str
        Output_filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    from osl_dynamics.data.task import epoch_mean

    epoched_1 = epoch_mean(data, time_index, pre, post)

    x_label = "Sample"
    time_index = np.arange(-pre, post)
    if sampling_frequency:
        time_index = time_index / sampling_frequency
        x_label = "Time (s)"

    # Validation
    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (16, 3)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Create figure
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)

    # Baseline correct
    if baseline_correct:
        epoched_1 -= np.mean(epoched_1[:pre], axis=0, keepdims=True)

    # Plot data
    for i, s in enumerate(epoched_1.T):
        ax.plot(time_index, s, label=i, **plot_kwargs)
    ax.axvline(0, c="k")
    ax.autoscale(axis="x", tight=True)

    # Set title and axis labels
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label)

    # Add a legend
    if legend:
        ax.legend(loc=legend_loc)

    # Save the figure if a filename has been passed
    if filename is not None:
        save(fig, filename)

    return fig, ax


def plot_matrices(
    matrix,
    group_color_scale=True,
    titles=None,
    main_title=None,
    cmap="viridis",
    nan_color="white",
    log_norm=False,
    filename=None,
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
        Matplotlib color to use for NaN values.
    log_norm: bool
        Should we show the elements on a log scale?
    filename: str
        A file to which to save the figure.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
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
            v_min = np.nanmin(matrix)
            v_max = np.nanmax(matrix)
            if log_norm:
                im = axis.matshow(
                    grid,
                    cmap=cmap,
                    norm=matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max),
                )
            else:
                im = axis.matshow(grid, vmin=v_min, vmax=v_max, cmap=cmap)
        else:
            if log_norm:
                im = axis.matshow(grid, cmap=cmap, norm=matplotlib.colors.LogNorm())
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

    if filename is not None:
        save(fig, filename, tight_layout=False)

    return fig, axes


def plot_connections(
    weights,
    labels=None,
    ax=None,
    cmap="hot",
    text_color=None,
    filename=None,
):
    """Create a chord diagram representing the values of a matrix.

    For a matrix of weights, create a chord diagram where the color of the line
    connecting two nodes represents the value indexed by the position of the nodes in
    the lower triangle of the matrix.

    This is useful for showing things like co-activation between sensors/parcels or
    relations between nodes in a network.

    Parameters
    ----------
    weights : numpy.ndarray
        An NxN matrix of weights.
    labels : list of str
        A name for each node in the weights matrix (e.g. parcel names)
    ax : matplotlib.pyplot.Axes
        A matplotlib axis on which to plot.
    cmap : str
        A string corresponding to a matplotlib colormap.
    text_color : str
        A string corresponding to a matplotlib color.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
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

    if filename is not None:
        save(fig, filename)

    return fig, ax


def topoplot(
    layout,
    data,
    channel_names=None,
    plot_boxes=False,
    show_deleted_sensors=False,
    show_names=False,
    title=None,
    colorbar=True,
    axis=None,
    cmap="plasma",
    n_contours=10,
    filename=None,
):
    """Make a contour plot in sensor space.

    Create a contour plot by interpolating a field from a set of values provided for
    each sensor location in an MEG layout. Within the context of DyNeMo this is likely
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
        matplotlib colourmap.
    n_contours: int
        number of field isolines to show on the plot.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
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
        save(fig, filename)

    return fig


def plot_brain_surface(
    values,
    mask_file,
    parcellation_file,
    filename=None,
    subtract_mean=False,
    mean_weights=None,
    **plot_kwargs,
):
    """Plot a 2D heat map on the surface of the brain.

    Parameters
    ----------
    values : np.ndarray
        Data to plot. Can be of shape: (n_maps, n_channels) or (n_channels,).
        A (..., n_channels, n_channels) array can also be passed.
        Warning: this function cannot be used if n_maps is equal to n_channels.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Output filename. If extension is .nii.gz the power map is saved as a
        NIFTI file. Or if the extension is png/svg/pdf, it is saved as images.
        Optional, if None is passed then the image is shown on screen.
    subtract_mean : bool
        Should we subtract the mean power across modes?
    mean_weights: np.ndarray
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    plot_kwargs : dict
        Keyword arguments to pass to nilearn.plotting.plot_img_on_surf.
    """
    from osl_dynamics.analysis import power

    power.save(
        power_map=values,
        filename=filename,
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
        **plot_kwargs,
    )


def plot_alpha(
    *alpha,
    n_samples=None,
    cmap="Set3",
    sampling_frequency=None,
    y_labels=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
):
    """Plot alpha.

    Parameters
    ----------
    alpha : numpy.ndarray
        A collection of alphas passed as separate arguments.
    n_samples: int
        Number of time points to be plotted.
    cmap : str
        A matplotlib colormap string.
    sampling_frequency : float
        The sampling frequency of the data in Hertz.
    y_labels : str
        Labels for the y-axis of each alpha time series.
    title : str
        Title for the plot.
    plot_kwargs : dict
        Any parameters to be passed to matplotlib.pyplot.stackplot.
    fig_kwargs : dict
        Arguments to pass to matplotlib.pyplot.subplots.
    filename : str
        Output filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    n_alphas = len(alpha)
    n_modes = max(a.shape[1] for a in alpha)
    n_samples = min(n_samples or np.inf, alpha[0].shape[0])
    if cmap in QUAL_CMAPS:
        cmap = plt.cm.get_cmap(name=cmap)
    else:
        cmap = plt.cm.get_cmap(name=cmap, lut=n_modes)
    colors = cmap.colors

    # Validation
    default_fig_kwargs = dict(
        figsize=(12, 2.5 * n_alphas), sharex="all", facecolor="white"
    )
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    default_plot_kwargs = dict(colors=colors)
    if plot_kwargs is None:
        plot_kwargs = default_plot_kwargs
    else:
        plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    if y_labels is None:
        y_labels = [None] * n_alphas
    elif isinstance(y_labels, str):
        y_labels = [y_labels] * n_alphas
    elif len(y_labels) != n_alphas:
        raise ValueError("Incorrect number of y_labels passed.")

    # Create figure
    fig, axes = create_figure(n_alphas, **fig_kwargs)

    # If n_alphas is one then axes won't be iterable
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot data
    for a, ax, y_label in zip(alpha, axes, y_labels):
        time_vector = (
            np.arange(n_samples) / sampling_frequency
            if sampling_frequency
            else range(n_samples)
        )
        ax.stackplot(time_vector, a[:n_samples].T, **plot_kwargs)
        ax.autoscale(tight=True)
        ax.set_ylabel(y_label)

    # Set axis label and title
    axes[-1].set_xlabel("Time (s)" if sampling_frequency else "Sample")
    axes[0].set_title(title)

    # Fix layout
    plt.tight_layout()

    # Add a colour bar
    norm = matplotlib.colors.BoundaryNorm(
        boundaries=range(n_modes + 1), ncolors=n_modes
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(right=0.94)
    cb_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    cb = fig.colorbar(mappable, cax=cb_ax, ticks=np.arange(0.5, n_modes, 1))
    cb.ax.set_yticklabels(range(1, n_modes + 1))

    # Save to file if a filename as been passed
    if filename is not None:
        save(fig, filename, tight_layout=False)

    return fig, axes


def plot_mode_lifetimes(
    mode_time_course,
    bins="auto",
    density=False,
    match_scale_x=False,
    match_scale_y=False,
    x_range=None,
    x_label=None,
    y_label=None,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
):
    """Create a histogram of mode lifetimes.

    For a mode time course, create a histogram for each mode with the distribution
    of the lengths of time for which it is active.

    Parameters
    ----------
    mode_time_course : numpy.ndarray
        Mode time course to analyse.
    bins : int
        Number of bins for the histograms.
    density : bool
        If True, plot the probability density of the mode activation lengths.
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
    plot_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.hist.
    fig_kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.subplots.
    filename : str
        A file to which to save the figure.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    from osl_dynamics.inference import modes

    n_plots = mode_time_course.shape[1]
    short, long, empty = rough_square_axes(n_plots)
    colors = get_colors(n_plots)

    # Validation
    if mode_time_course.ndim == 1:
        mode_time_course = get_one_hot(mode_time_course)
    if mode_time_course.ndim != 2:
        raise ValueError("mode_timecourse must be a 2D array")

    default_fig_kwargs = {"figsize": (long * 2.5, short * 2.5)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Calculate mode lifetimes
    channel_lifetimes = modes.lifetimes(mode_time_course)

    # Create figure
    fig, axes = create_figure(short, long, **fig_kwargs)

    # Plot data
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
            channel, density=density, bins=bins, color=color, **plot_kwargs
        )
        largest_bar = max(hist[0].max(), largest_bar)
        furthest_value = max(hist[1].max(), furthest_value)
        t = axis.text(
            0.95,
            0.95,
            f"{np.sum(channel) / len(mode_time_course) * 100:.2f}%",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
        )
        axis.xaxis.set_tick_params(labelbottom=True, labelleft=True)
        t.set_bbox({"facecolor": "white", "alpha": 0.7, "boxstyle": "round"})

    # Set axis range and labels
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

    # Save file is a filename has been passed
    if filename is not None:
        save(fig, filename)

    return fig, axes
