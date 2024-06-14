"""Temporal Interval Network Density Analysis (TINDA).

This module contains functions for calculating the density profile (i.e.,
fractional occupancy over) in any interval between events it is originally
intended to use it on an HMM state time course to ask questions like what is
the density of state j in the first and second part of the interval between
visits to state i.

See Also
--------
`Example script <https://github.com/OHBA-analysis/osl-dynamics/blob/main\
/examples/simulation/hmm_tinda.py>`_ applying TINDA to simulated HMM data.
"""

from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def find_intervals(tc_hot):
    """Find intervals (periods where :code:`tc_hot` is zero) in a hot vector.

    Parameters
    ----------
    tc_hot : array_like
        Hot vector (i.e., binary vector) of shape (n_samples,) or
        (n_samples, 1). For example, a hot vector of a state time course
        of shape (n_samples, n_states).

    Returns
    -------
    intervals : list
        List of tuples of start and end indices of intervals.
    durations : array_like
        Array of durations of intervals (in samples).
    """
    intervals = []
    durations = []
    tc_tmp = np.insert(np.insert(tc_hot, 0, 0, axis=0), -1, 0, axis=0)
    start = np.where(np.diff(tc_tmp) == 1)[0]
    end = np.where(np.diff(tc_tmp) == -1)[0]
    intervals = list(zip(end[:-1], start[1:]))
    durations = np.diff(intervals, axis=1).squeeze()
    return intervals, durations


def split_intervals(intervals, n_bins=2):
    """Splits each interval into nbin equally sized bins.

    Parameters
    ----------
    intervals : list
        List of tuples of start and end indices of intervals.
    n_bins : int, optional
        Number of bins to split each interval into.

    Returns
    -------
    divided_intervals : list
        List the same length as intervals (minus dropped intervals, see below),
        with each element being a list of tuples of start and end indices of
        bins.
    bin_sizes : list
        List of bin sizes (in samples), one per interval.
    drop_mask : array_like
        Array of zeros and ones indicating whether the interval was dropped
        because it was smaller than :code:`n_bins`.
    """
    divided_intervals = []
    bin_sizes = []
    drop_mask = np.zeros(len(intervals))
    for i, (interval_start, interval_end) in enumerate(intervals):
        n_samples = interval_end - interval_start
        if n_samples < n_bins:
            drop_mask[i] = 1
            continue

        bin_size = n_samples // n_bins
        remainder = n_samples % n_bins
        bins = []
        bin_start = interval_start
        for i in range(n_bins):
            bin_end = bin_start + bin_size
            bins.append((bin_start, bin_end))
            if remainder > 0:
                if remainder == n_bins - 1:
                    bin_end += 1
                elif i == (n_bins // 2 - 1):
                    bin_end += remainder
            bin_start = bin_end
        divided_intervals.append(bins)
        bin_sizes.append(bin_size)

    return divided_intervals, bin_sizes, drop_mask


def split_interval_duration(
    durations, interval_range=None, mode="sample", sampling_frequency=None
):
    """Split interval durations into bins based on their duration.

    Parameters
    ----------
    durations : array_like
        Array of durations of intervals (in samples).
    interval_range : array_like, optional
        Array of bin edges (in samples, seconds, or percentiles) to split
        durations into bins are defined as :code:`[>=interval_range[i],
        <interval_range[i+1])`. If :code:`None`, all durations are in the
        same bin.
    mode : str, optional
        Mode of interval_range, either :code:`"sample"` (e.g.,
        :code:`[4, 20, 100]`), :code:`"perc"` (e.g., :code:`range(20,100,20)`),
        or :code:`"sec"` (e.g., :code:`[0, 0.01, 0.1, 1, np.inf]`). If
        :code:`"sec"`, :code:`sfreq` must be provided.
    sampling_frequency : float, optional
        Sampling frequency (in Hz) of the data, only used if :code:`mode` is
        :code:`"sec"`.

    Returns
    -------
    mask : list
        List of arrays of zeros and ones indicating whether the interval was
        in the bin.
    interval_range : array_like
        Array of bin edges (in samples) used to split durations into bins.
    """
    if interval_range is None:
        mask = [np.ones_like(durations)]
        interval_range = [np.min(durations), np.max(durations)]
    else:
        if mode == "sec" and sampling_frequency is None:
            raise ValueError(
                "Sampling frequency (sfreq) must be specified when mode is 'sec'"
            )
            return [np.ones_like(durations)], None
        else:
            if mode == "sec":
                interval_range = [r * sampling_frequency for r in interval_range]
            elif mode == "perc":
                interval_range = np.percentile(durations, interval_range)
            mask = []
            for i, start in enumerate(interval_range):
                if i < len(interval_range) - 1:
                    hot_vector = np.logical_and(
                        durations >= start, durations < interval_range[i + 1]
                    )
                    mask.append(hot_vector.astype(int))
    return mask, interval_range


def compute_fo_stats(
    tc_sec, divided_intervals, interval_mask=None, return_all_intervals=False
):
    """Compute sums and weighted averages of time courses in each interval.

    Parameters
    ----------
    tc_sec : array_like
        Time course of shape (n_samples, n_states).
    divided_intervals : list
        List with each element corresponding to an interval, each itself being a
        list of tuples of start and end indices of interval bins.
    interval_mask : array_like, optional
        Array of zeros and ones indicating whether the interval was in the bin.
    return_all_intervals : bool, optional
        Whether to return the density/sum of all intervals in addition to the
        interval averages/sums.

    Returns
    -------
    interval_weighted_avg : array_like
        Array of weighted averages of time courses in each interval of shape
        (n_states, n_bins, n_interval_ranges).
    interval_sum : array_like
        Array of sums of time courses in each interval of shape (n_states,
        n_bins, n_interval_ranges).
    interval_weighted_avg_all : list
        List of length n_interval_ranges with each element an array of weighted
        averages of time courses in each interval of shape (n_states, n_bins,
        n_intervals). :code:`None` if :code:`return_all_intervals=False`
        (default).
    interval_sum_all : list
        List of length :code:`n_interval_ranges` with each element an array of
        sums of time courses in each interval of shape (n_states, n_bins,
        n_intervals). :code:`None` if :code:`return_all_intervals=False`.
    """
    if interval_mask is None:
        interval_mask = [np.ones(len(divided_intervals))]

    if (
        len(divided_intervals[0]) == 2
    ):  # this corresponds to the matlab code but only works for two bins
        # and I think it's less principled than the code below
        intervals = []
        for interval in divided_intervals:
            intervals.append([interval[0][0], interval[-1][-1]])

        interval_weighted_avg = np.zeros((tc_sec.shape[1], 2, len(interval_mask)))
        interval_sum = np.zeros((tc_sec.shape[1], 2, len(interval_mask)))
        temp_to = []
        temp_away = []
        for i in intervals:
            d = int(np.floor((np.diff(i) - 1) / 2))
            temp_away.append(tc_sec[i[0] : i[0] + d + 1, :])
            temp_to.append(tc_sec[i[1] - d - 1 : i[1], :])

        interval_weighted_avg_all = []
        interval_sum_all = []
        for j, mask in enumerate(interval_mask):
            temp_to_flat = np.concatenate([temp_to[k] for k in np.where(mask == 1)[0]])
            temp_away_flat = np.concatenate(
                [temp_away[k] for k in np.where(mask == 1)[0]]
            )

            interval_weighted_avg[:, 0, j] = np.mean(temp_away_flat, axis=0)
            interval_weighted_avg[:, 1, j] = np.mean(temp_to_flat, axis=0)
            interval_sum[:, 0, j] = np.sum(temp_away_flat, axis=0)
            interval_sum[:, 1, j] = np.sum(temp_to_flat, axis=0)
            interval_weighted_avg_all.append(
                np.transpose(
                    np.stack(
                        [
                            np.stack(
                                [
                                    temp_away[k].mean(axis=0)
                                    for k in np.where(mask == 1)[0]
                                ],
                                axis=-1,
                            ),
                            np.stack(
                                [
                                    temp_to[k].mean(axis=0)
                                    for k in np.where(mask == 1)[0]
                                ],
                                axis=-1,
                            ),
                        ]
                    ),
                    axes=[1, 0, 2],
                )
            )
            interval_sum_all.append(
                np.transpose(
                    np.stack(
                        [
                            np.stack(
                                [
                                    temp_away[k].sum(axis=0)
                                    for k in np.where(mask == 1)[0]
                                ],
                                axis=-1,
                            ),
                            np.stack(
                                [
                                    temp_to[k].sum(axis=0)
                                    for k in np.where(mask == 1)[0]
                                ],
                                axis=-1,
                            ),
                        ]
                    ),
                    axes=[1, 0, 2],
                )
            )
    else:
        # TODO: I think this is more principled than the matlab code,
        # but I need to check
        interval_sum = np.zeros(
            (tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals))
        )
        interval_weighted_avg = np.zeros(
            (tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals))
        )
        for i, interval in enumerate(divided_intervals):
            for j, (start, end) in enumerate(interval):
                interval_sum[:, j, i] = np.sum(tc_sec[start:end, :], axis=0)
            interval_weighted_avg[:, :, i] = interval_sum[:, :, i] / (end - start)

        interval_weighted_avg_all = [
            interval_weighted_avg[:, :, interval_selection == 1]
            for interval_selection in interval_mask
        ]
        interval_sum_all = [
            interval_sum[:, :, interval_selection == 1]
            for interval_selection in interval_mask
        ]
        interval_weighted_avg = np.stack(
            [weighted_avg.mean(axis=-1) for weighted_avg in interval_weighted_avg_all],
            axis=-1,
        )
        interval_sum = np.stack(
            [int_sum.mean(axis=-1) for int_sum in interval_sum_all], axis=-1
        )

    if return_all_intervals:
        return (
            interval_weighted_avg,
            interval_sum,
            interval_weighted_avg_all,
            interval_sum_all,
        )
    else:
        return interval_weighted_avg, interval_sum, None, None


def collate_stats(stats, field, all_to_all=False, ignore_elements=None):
    """Collate list of stats (e.g., of different states) into a single array.

    Parameters
    ----------
    stats : list
        List of stats (:code:`dict`) for each state. Each element is a
        dictionary with keys that at least should include "field" (e.g.,
        :code:`interval_wavg`), that is the output of :code:`compute_fo_stats`.
    field : str
        Field of stats to collate, e.g., :code:`"interval_wavg"`,
        :code:`"interval_sum"`.
    all_to_all : bool, optional
        Whether the density_of was used to compute the stats (in which case
        the first 2 dimensions are not :code:`n_states` x :code:`n_states`).
        Default is :code:`False`.
    ignore_elements : list, optional
        List of indices in stats to ignore (i.e. because they don't contain
        binary events).

    Returns
    -------
    collated_stat : array_like
        The collated stat (:code:`n_interval_states`, :code:`n_density_states`,
        :code:`n_bins`, :code:`n_interval_ranges`). If :code:`all_to_all=False`
        (default) (i.e., when the density is computed for all states using all
        states' intervals), then the first two dimensions are :code:`n_states`
        and the diagonal is :code:`np.nan`.
    """
    if ignore_elements is None:
        ignore_elements = []

    num_states = len(stats)
    shp = stats[0][field].shape  # (n_states, n_bins, n_interval_ranges)
    if num_states > 1:
        if all_to_all:
            collated_stat = np.full((2 * [num_states] + list(shp[1:])), np.nan)
        else:
            collated_stat = np.full(([num_states] + list(shp)), np.nan)

        for i in range(num_states):
            if i in ignore_elements:
                # intervals are not binary, keep a row of nans
                continue
            if all_to_all:
                collated_stat[i, np.arange(num_states) != i, ...] = stats[i][field]
            else:
                collated_stat[i] = stats[i][field]

    else:
        collated_stat = stats[0][field]

    return collated_stat


def tinda(
    tc,
    density_of=None,
    n_bins=2,
    interval_mode=None,
    interval_range=None,
    sampling_frequency=None,
    return_all_intervals=False,
):
    """Compute time-in-state density and sum for each interval.

    Parameters
    ----------
    tc : array_like
        Time courses of shape (n_samples, n_states) define intervals from will
        use the same time courses to compute density of when :code:`density_of`
        is :code:`None`. Can be a list of time courses (e.g. state time courses
        for each individual).
    density_of : array_like, optional
        Time course of shape (n_samples, n_states) to compute density of if
        :code:`None` (default), density is computed for all columns of tc.
    n_bins : int, optional
        Number of bins to divide each interval into (default 2).
    interval_mode : str, optional
        Mode of :code:`interval_range`, either :code:`"sample"` (default),
        "sec" (seconds) or "perc" (percentile). To interpret the interval range
        as seconds, :code:`sfreq` must be provided.
    interval_range : array_like, optional
        Array of bin edges (in samples, seconds, or percentiles) used to split
        durations into bins (default :code:`None`), e.g.
        :code:`np.arange(0, 1, 0.1)` for 100 ms bins.
    sampling_frequency : float, optional
        Sampling frequency of tc (in Hz), only used if
        :code:`interval_mode="sec"`.
    return_all_intervals : bool, optional
        Whether to return the density/sum of all intervals in addition to the
        interval averages/sums. If :code:`True`, will return a list of arrays in
        :code:`stats[i]['all_interval_wavg'/'all_interval_sum']`, each
        corresponding to an interval range.

    Returns
    -------
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, n_bins, n_interval_ranges). :code:`n_interval_states`
        is the number of states in the interval time courses (i.e., tc);
        :code:`n_density_states` is the number of states in the density time
        courses (i.e., :code:`density_of`). If :code:`density_of` is
        :code:`None`, :code:`n_density_states` is the same as
        :code:`n_interval_states`. If tc is a list of time courses (e.g., state
        time courses for multiple individuals), then an extra dimension is appended
        for the individuals.
    fo_sum : array_like
        Same as :code:`fo_density`, but with time-in-state sums instead of
        densities.
    stats : dict
        Dictionary of stats, including

        - :code:`durations`: interval durations in samples.
        - :code:`intervals`: start/end samples for each interval (intervals).
        - :code:`interval_wavg`: the weighted average (i.e, time-in-state
          density) over all interval.
        - :code:`interval_sum`: the sum (i.e., time-in-state) over all
          intervals.
        - :code:`divided_intervals`: the bin edges for each interval.
        - :code:`bin_sizes`: the bin sizes for each interval.
        - :code:`interval_range`: the interval range (in samples).
        - :code:`all_interval_wavg`: unaveraged interval densities (only if
          :code:`return_all_intervals=True`).
        - :code:`all_interval_sum`: unaveraged interval sums (only if
          :code:`return_all_intervals=True`).
    """
    if isinstance(
        tc, list
    ):  # list of time courses (e.g., individuals' HMM state time courses)
        if density_of is None:
            fo_density_tmp, fo_sum_tmp, stats = zip(
                *[
                    tinda(
                        itc,
                        None,
                        n_bins,
                        interval_mode,
                        interval_range,
                        sampling_frequency,
                        return_all_intervals,
                    )
                    for itc in tc
                ]
            )
        elif len(density_of) == len(tc):
            fo_density_tmp, fo_sum_tmp, stats = zip(
                *[
                    tinda(
                        itc,
                        density_of[ix],
                        n_bins,
                        interval_mode,
                        interval_range,
                        sampling_frequency,
                        return_all_intervals,
                    )
                    for ix, itc in enumerate(tc)
                ]
            )
        fo_density = np.stack(fo_density_tmp, axis=-1)
        fo_sum = np.stack(fo_sum_tmp, axis=-1)

    else:
        stats = []
        dim = tc.shape
        ignore_elements = []

        for i in range(dim[1]):
            itc_prim = tc[:, i]
            if not np.array_equal(
                itc_prim, itc_prim.astype(int)
            ):  # if not binary (i.e., intervals are not well difined)
                stats.append(None)
                ignore_elements.append(i)

            else:
                if density_of is None:
                    # we're doing density of all states in all states' intervals
                    itc_sec = tc[:, np.setdiff1d(range(dim[1]), i)]
                else:
                    itc_sec = density_of

                # get interval info
                intervals, durations = find_intervals(itc_prim)
                divided_intervals, bin_sizes, dropped_intervals = split_intervals(
                    intervals, n_bins
                )  # split intervals into nbin
                durations = durations[
                    dropped_intervals == 0
                ]  # drop intervals that are too short to be split into nbin
                interval_mask, interval_range_samples = split_interval_duration(
                    durations,
                    interval_range=interval_range,
                    mode=interval_mode,
                    sampling_frequency=sampling_frequency,
                )  # split intervals into interval_range (i.e.,
                # to compute statistics of intervals with durations in a
                # certain range)

                # Compute time-in-state densities and sums in all intervals
                (
                    interval_wavg,
                    interval_sum,
                    all_interval_wavg,
                    all_interval_sum,
                ) = compute_fo_stats(
                    itc_sec,
                    divided_intervals,
                    interval_mask,
                    return_all_intervals=return_all_intervals,
                )

                # Append stats
                stats.append(
                    {
                        "durations": durations,
                        "intervals": intervals,
                        "interval_wavg": interval_wavg,
                        "interval_sum": interval_sum,
                        "divided_intervals": divided_intervals,
                        "bin_sizes": bin_sizes,
                        "interval_range": interval_range_samples,
                        "all_interval_wavg": all_interval_wavg,
                        "all_interval_sum": all_interval_sum,
                    }
                )

        # Get a full matrix of FO densities and sums
        fo_density = collate_stats(
            stats,
            "interval_wavg",
            all_to_all=density_of is None,
            ignore_elements=ignore_elements,
        )
        fo_sum = collate_stats(
            stats,
            "interval_sum",
            all_to_all=density_of is None,
            ignore_elements=ignore_elements,
        )

    return fo_density, fo_sum, stats


def circle_angles(order):
    """Compute the phase differences between states in a circular plot.

    Parameters
    ----------
    order : list
        List of state orders (in order of counterclockwise
        rotation).

    Returns
    -------
    angleplot : array_like
        Array of phase differences between states in a circular plot.

    """
    K = len(order)
    disttoplot_manual = np.zeros(K, dtype=complex)
    for i3 in range(K):
        disttoplot_manual[order[i3]] = np.exp(1j * (i3 + 1) / K * 2 * np.pi)

    angleplot = np.exp(
        1j
        * (
            np.angle(disttoplot_manual[:, np.newaxis]).T
            - np.angle(disttoplot_manual[:, np.newaxis])
        )
    )
    return angleplot


def optimise_sequence(fo_density, metric_to_use=0, n_perms=10**6):
    """Optimise the sequence to maximal circularity.

    This function reads in the mean pattern of differential fractional
    occupancy and computes the optimal display for a sequential circular plot
    visualization.

    Parameters
    ----------
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, 2, n_sessions).
    metric : int, optional
        Metric to use for optimisation:

        - :code:`0`: mean FO asymmetry.
        - :code:`1`: proportional FO asymmetry (i.e. asymmetry as a proportion
          of a baseline - which time spend in the state).
        - :code:`2`: proportional FO asymmetry using global baseline FO, rather
          than a individual-specific baseline.

    Returns
    -------
    best_sequence : list
        List of best sequence of states to plot (in order of counterclockwise
        rotation).
    """
    if len(fo_density.shape) == 5:
        fo_density = np.squeeze(fo_density)

    # make sure there are no nans:
    fo_density[np.isnan(fo_density)] = 0

    # Compute different metrics to optimise
    metric = []
    metric.append(np.mean(fo_density[:, :, 0, :] - fo_density[:, :, 1, :], axis=2))
    temp = (fo_density[:, :, 0, :] - fo_density[:, :, 1, :]) / np.mean(
        fo_density, axis=2
    )
    temp[np.isnan(temp)] = 0
    metric.append(np.mean(temp, axis=2))
    metric.append(
        np.mean(fo_density[:, :, 0, :] - fo_density[:, :, 1, :], axis=2)
        / np.mean(fo_density, axis=(2, 3))
    )
    n_metrics = len(metric)
    K = fo_density.shape[0]

    best_sequence = []
    for i in range(n_metrics):
        ix = np.arange(K)
        v = np.imag(np.sum(circle_angles(ix) * metric[i]))
        cnt = 0
        while cnt < n_perms:
            cnt += 1
            swaps = np.random.permutation(K)
            swaps = swaps[:2]
            tmpix = ix.copy()
            tmpix[swaps[0]] = ix[swaps[1]]
            tmpix[swaps[1]] = ix[swaps[0]]
            tmpv = np.imag(np.sum(circle_angles(tmpix) * metric[i]))
            if tmpv < v:
                v = tmpv
                ix = tmpix
        best_sequence.append(np.roll(ix, -np.where([iix == 0 for iix in ix])[0][0]))
    # Return the best sequence for the chosen metric (in order of counterclockwise
    # rotation)
    return best_sequence[metric_to_use]


def compute_cycle_strength(angleplot, asym, relative=True, whichstate=None):
    if len(asym.shape) == 3:
        tmp = np.stack(
            [angleplot * asym[:, :, i] for i in range(asym.shape[2])], axis=-1
        )
    else:
        tmp = angleplot * asym
    if whichstate is not None:
        # Note that we are counting each (i,j) double because for the rotational
        # momentum per state we take into account (i,j) and (j,i) for all j and one
        # particular i.
        tmp = np.squeeze(
            tmp[
                whichstate,
                :,
            ]
        ) + np.squeeze(tmp[:, whichstate])
        cycle_strength = np.imag(np.nansum(tmp, axis=0))
    else:
        cycle_strength = np.imag(np.nansum(tmp, axis=(0, 1)))

    # positive rotational momentum should indicate clockwise cycle
    cycle_strength = -cycle_strength

    if relative:  # normalise by the theoretical maximum
        cycle_strength = cycle_strength / np.abs(
            compute_cycle_strength(
                angleplot,
                np.sign(np.imag(angleplot)),
                relative=False,
                whichstate=whichstate,
            )
        )

    return cycle_strength


def plot_cycle(
    ordering,
    fo_density,
    edges,
    new_figure=False,
    color_scheme=None,
):
    """Plot state network as circular diagram with arrows.

    Parameters
    ----------
    ordering : list
        List of best sequence of states to plot (in order of counterclockwise
        rotation).
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, 2, (n_interval_ranges,) n_sessions).
    edges : array_like
        Array of zeros and ones indicating whether the connection should be
        plotted.
    new_figure : bool, optional
        Whether to create a new figure (default is :code:`False`).
    color_scheme : array_like, optional
        Array of size (K,3) color scheme to use for plotting (default is
        :code:`None`). If :code:`None`, will use the default color scheme from
        the matlab code.
    """

    # Plot state network as circular diagram with arrows
    if color_scheme is None:
        color_scheme = np.array(
            [
                [0, 0, 1.0000],
                [1.0000, 0.3333, 0],
                [
                    1.0000,
                    0.6667,
                    0,
                ],
                [
                    0.6667,
                    1.0000,
                    0.3333,
                ],
                [
                    0.3333,
                    1.0000,
                    0.6667,
                ],
                [
                    0,
                    1.0000,
                    1.0000,
                ],
                [
                    0.5529,
                    0.8275,
                    0.7804,
                ],
                [
                    1.0000,
                    0.5000,
                    0.5000,
                ],
                [
                    0,
                    0.6667,
                    1.0000,
                ],
                [
                    1.0000,
                    1.0000,
                    0,
                ],
                [
                    0.7451,
                    0.7294,
                    0.8549,
                ],
                [
                    0.6667,
                    0,
                    0,
                ],
            ]
        )
    if new_figure:
        plt.figure(figsize=(6.02, 4.52), dpi=100)
    else:
        plt.gca()

    K = len(ordering)
    if len(fo_density.shape) == 5:
        fo_density = np.squeeze(
            fo_density
        )  # squeeze in case there is still a interval_ranges dimension

    # compute mean direction of arrows
    mean_direction = np.squeeze(
        (fo_density[:, :, 0, :] - fo_density[:, :, 1, :]).mean(axis=2)
    )

    # reorder the states to match the ordering:
    ordering = np.roll(
        ordering[::-1], 1
    )  # rotate ordering from clockwise to counter clockwise
    edges = edges[ordering][:, ordering]
    mean_direction = mean_direction[ordering][:, ordering]

    # get the locations on the unit circle
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / K)
    x = np.roll(np.cos(theta), int(K / 4))  # start from 12 o'clock
    y = np.roll(np.sin(theta), int(K - (K / 4)))
    distance_to_plot_manual = np.stack([x, y]).T

    # plot the scatter points with state identities
    for i in range(K):
        plt.scatter(
            distance_to_plot_manual[i, 0],
            distance_to_plot_manual[i, 1],
            s=400,
            color=color_scheme[ordering[i], :],
        )
        plt.text(
            distance_to_plot_manual[i, 0],
            distance_to_plot_manual[i, 1],
            str(ordering[i] + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
        )

    # plot the arrows
    for ik1 in range(K):
        for k2 in range(K):
            if edges[ik1, k2]:
                # arrow lengths have to be proportional to the distance
                # between the states. Use Pythagoras:
                line_scale = np.sqrt(
                    np.sum(
                        (
                            distance_to_plot_manual[k2, :]
                            - distance_to_plot_manual[ik1, :]
                        )
                        ** 2
                    )
                )
                arrow_start = (
                    distance_to_plot_manual[ik1, :]
                    + 0.1
                    * (distance_to_plot_manual[k2, :] - distance_to_plot_manual[ik1, :])
                    / line_scale
                )
                arrow_end = (
                    distance_to_plot_manual[k2, :]
                    - 0.1
                    * (distance_to_plot_manual[k2, :] - distance_to_plot_manual[ik1, :])
                    / line_scale
                )
                if mean_direction[ik1, k2] > 0:  # arrow from k1 to k2:
                    plt.arrow(
                        arrow_start[0],
                        arrow_start[1],
                        arrow_end[0] - arrow_start[0],
                        arrow_end[1] - arrow_start[1],
                        head_width=0.05,
                        head_length=0.1,
                        length_includes_head=True,
                        color="k",
                    )
                elif mean_direction[ik1, k2] < 0:  # arrow from k2 to k1:
                    plt.arrow(
                        arrow_end[0],
                        arrow_end[1],
                        arrow_start[0] - arrow_end[0],
                        arrow_start[1] - arrow_end[1],
                        head_width=0.05,
                        head_length=0.1,
                        length_includes_head=True,
                        color="k",
                    )

    plt.axis("off")
    plt.axis("equal")
