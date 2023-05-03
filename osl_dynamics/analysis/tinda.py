"""
Temporal Interval Network Density Analysis (TINDA)
This module contains functions for calculating the density profile 
(i.e., fractional occupancy over) in any interval between events
it is originally intended to use it on an HMM state time course to 
ask questions like what is the density of state j in the first and 
second part of the interval between visits to state i
"""


import numpy as np
from osl_dynamics.inference.modes import argmax_time_courses



def find_intervals(tc_hot):
    """Find intervals (periods where tc_hot is zero) in a hot vector

    Parameters
    ----------
    tc_hot: array_like
        hot vector (i.e., binary vector) of shape (n_samples,) or (n_samples, 1)
        For example, a hot vector of a state time course of shape (n_samples, n_states)

    Returns
    -------
    intervals: list
        list of tuples of start and end indices of intervals
    durations: array_like
        array of durations of intervals (in samples)

    """
    intervals = []
    durations = []
    tc_tmp = np.insert(np.insert(tc_hot, 0, 0, axis=0), -1, 0, axis=0)
    start = np.where(np.diff(tc_tmp)==1)[0]
    end = np.where(np.diff(tc_tmp)==-1)[0]
    intervals = list(zip(end[:-1], start[1:]))
    durations = np.diff(intervals, axis=1).squeeze()
    return intervals, durations


def split_intervals(intervals, nbin=2):
    """Split each interval into nbin equal sized bins

    Parameters
    ----------
    intervals: list
        list of tuples of start and end indices of intervals
    nbin: int
        number of bins to split each interval into

    Returns
    -------
    divided_intervals: list
        list the same length as intervals (minus dropped intervals, see below), 
        with each element being a list of tuples of start and end indices of bins
    bin_sizes: list
        list of bin sizes (in samples), one per interval
    drop_mask: array_like
        array of zeros and ones indicating whether the interval was dropped because it was smaller than nbin
    
    """
    divided_intervals = []
    bin_sizes = []
    drop_mask = np.zeros(len(intervals))
    for i, (start, end) in enumerate(intervals):
        n_samples = end - start
        bin_size = n_samples // nbin
        if bin_size > 0:
            remainder = n_samples % nbin
            bins = []
            bin_start = start
            for i in range(nbin):
                bin_end = bin_start + bin_size
                bins.append((bin_start, bin_end))
                if remainder > 0:
                    if remainder == nbin - 1:
                        bin_end += 1
                    elif i == (nbin //2 - 1):
                        bin_end += remainder
                bin_start = bin_end
            divided_intervals.append(bins)
            bin_sizes.append(bin_size)
        else:
            drop_mask[i] = 1
    return divided_intervals, bin_sizes, drop_mask


def split_interval_duration(durations, interval_range=None, mode="sample", sfreq=None):
    """Split interval durations into bins based on their duration

    Parameters
    ----------
    durations: array_like
        array of durations of intervals (in samples)
    interval_range: array_like
        array of bin edges (in samples, seconds, or percentiles) to split durations into
        bins are defined as [>=interval_range[i], <interval_range[i+1])
    mode: str
        mode of interval_range, either "sample" (e.g., [4, 20, 100]), "perc" (e.g., range(20,100,20)), 
        or "sec" (e.g., [0, 0.01, 0.1, 1, np.inf])
    sfreq: float
        sampling frequency (in Hz) of the data, only used if mode is "sec"

    Returns
    -------
    mask: list
        list of arrays of zeros and ones indicating whether the interval was in the bin
    interval_range: array_like
        array of bin edges (in samples) used to split durations into bins
    """
    if mode == "sec" and sfreq is None:
        raise ValueError("Sampling frequency (sfreq) must be specified when mode is 'sec'")
    if interval_range is None:
        return [np.ones_like(durations)], None
    else:
        if mode == "sec":
            interval_range = [r * sfreq for r in interval_range]
        elif mode == "perc":
            interval_range = np.percentile(durations, interval_range)
        mask = []
        for i in range(len(list(interval_range))-1):
            hot_vector = np.logical_and(durations >= interval_range[i], durations < interval_range[i+1])
            mask.append(hot_vector.astype(int))
        return mask, interval_range



def compute_fo_stats(tc_sec, divided_intervals, interval_mask=None, avg_intervals=True):
    """Compute weighted averages (weighted by interval duration and mean occurrence), 
        and sums, of time courses in each interval

    Parameters
    ----------
    tc_sec: array_like
        time course of shape (n_samples, n_states)
    divided_intervals: list
        list with each element corresponding to an interval, each itself being
        a list of tuples of start and end indices of interval bins
    interval_mask: array_like
        array of zeros and ones indicating whether the interval was in the bin
    avg_intervals: bool
        whether to average the intervals within each bin (True) or sum/average them (False)

    Returns
    -------
    interval_weighted_avg: array_like
        array of weighted averages of time courses in each interval of shape (n_states, n_bins, n_intervals)
        if avg_intervals is False, otherwise of shape (n_states, n_intervals)
    interval_sum: array_like    
        array of sums of time courses in each interval of shape (n_states, n_bins, n_intervals)
        if avg_intervals is False, otherwise of shape (n_states, n_intervals)    
    """
    if interval_mask is None:
        interval_mask  = [np.ones(len(divided_intervals))]
    tc_mean = np.expand_dims(tc_sec.mean(axis=0), axis=1)
    interval_sum = np.zeros((tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals)))
    interval_weighted_avg = np.zeros((tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals)))
    for i, interval in enumerate(divided_intervals):
        for j, (start, end) in enumerate(interval):
            interval_sum[:,j,i] = np.sum(tc_sec[start:end,:], axis=0)
        interval_weighted_avg[:,:,i] = tc_mean*(interval_sum[:,:,i] / (end - start))
    interval_weighted_avg = [interval_weighted_avg[:,:,interval_selection==1].mean(axis=2) if avg_intervals else interval_weighted_avg[:,:,interval_selection==1] for interval_selection in interval_mask]
    interval_sum = [interval_sum[:,:,interval_selection==1].sum(axis=2) if avg_intervals else interval_sum[:,:,interval_selection==1] for interval_selection in interval_mask]
    return interval_weighted_avg, interval_sum


def collate_stats(stats, field, ignore_elements=None):
    """Collate stats across states

    Parameters
    ----------
    stats: list
        list of stats (dict) for each state. Each element is a dictionary with keys
        that at least should include "field" (e.g., interval_wavgs), that is the output
        of compute_fo_stats
    field: str
        field of stats to collate, e.g., "interval_wavgs", "interval_sums"
    ignore_elements: list
        list of states to ignore (i.e. because they don't contain binary events)

    Returns
    -------
    collated_stat: list
        list of collated stats for each interval range
    """
    num_states = len(stats)
    if num_states > 1:
        nbins = stats[0][field][0].shape[-1]
        n_ranges = len(stats[0][field])
        collated_stat = [np.full((num_states, num_states, nbins), np.nan) for _ in range(n_ranges)]
        for i in range(num_states):
            if i in ignore_elements:
                continue
            for k in range(n_ranges):
                stat = stats[i][field][k]
                if len(stat.shape)>2:
                    stat = stat.mean(axis=2)
                collated_stat[k][i, np.setdiff1d(range(num_states),i)] = stat
    else:
        collated_stat = stats[0][field]
    return collated_stat


def tinda(tc, density_of=None, nbin=2, interval_mode=None, interval_range=None, sfreq=None, avg_intervals=True):
    """Compute time-in-state density and sum for each interval
    
    Parameters
    ---------- 
    tc: array_like
        time courses of shape (n_samples, n_states) define intervals from
        will use the same time courses to compute density of when density_of is None
        Can be a list of time courses (e.g. state time courses for each subject)
    density_of: array_like
        time course of shape (n_samples, n_states) to compute density of
        if None (default), density is computed for all columns of tc
    nbin: int
        number of bins to divide each interval into (default 2)
    interval_mode: str
        mode of interval_range, either "sample" (default), "sec" (seconds) or "perc" (percentile)
        To interpret the interval range as seconds, sfreq must be provided
    interval_range: array_like
        array of bin edges (in samples, seconds, or percentiles) 
        used to split durations into bins (default None)
        e.g. np.arange(0, 1, 0.1) for 100ms bins
    sfreq: float
        sampling frequency of tc (in Hz), only used if interval_mode is "sec"
    avg_intervals: bool
        whether to average/sum the intervals within each bin (True) keep the full dimensionality (False)

    Returns
    -------
    fo_density: list
        list with elements corresponding to interval ranges, each itself being an
        array of time-in-state densities of shape (n_states, n_states, n_bins, n_subjects)
        if avg_intervals is False, otherwise of shape (n_states, n_intervals)
        will return a list of arrays if 
    fo_sum: array_like
        same as fo_density, but with time-in-state sums instead of densities
    stats: dict
        dictionary of stats, including 
        - interval durations in samples (durations) 
        - start/end samples for each interval (intervals) 
        - the weighted average (i.e, time-in-state density) over all intervals (interval_wavg)
        - the sum (i.e., time-in-state) over all intervals (interval_sum)
        - the bin edges (divided_intervals) for each interval
        - the bin sizes (bin_sizes) for each interval
        - the interval range (in samples)
    """
    if isinstance(tc, list): # list of time courses (e.g., subjects' HMM state time courses)
        fo_density_tmp, fo_sum_tmp, stats = zip(*[tinda(itc, density_of[ix], nbin, interval_mode, interval_range, sfreq, avg_intervals) for ix, itc in enumerate(tc)])
        # TODO: FIX!
        s = fo_density_tmp[0][0].shape
        n_ranges = len(fo_density_tmp[0])
        n_sub = len(fo_density_tmp)
        fo_density = [np.stack([fo_density_tmp[j][i] for j in range(n_sub)], axis=3) for i in range(n_ranges)]
        fo_sum = [np.stack([fo_density_tmp[j][i] for j in range(n_sub)], axis=3) for i in range(n_ranges)]
    else:
        stats=[]
        dim = tc.shape
        ignore_elements = []
        for i in range(dim[1]):
            itc_prim = tc[:, i]
            if not np.array_equal(itc_prim, itc_prim.astype(int)): # if not binary (i.e., intervals are not well difined)
                stats.append(None)
                ignore_elements.append(i)
            else:
                if density_of is None:
                    itc_sec = tc[:, np.setdiff1d(range(dim[1]), i)]
                else:
                    itc_sec = density_of
                intervals, durations = find_intervals(itc_prim)
                divided_intervals, bin_sizes, dropped_intervals = split_intervals(intervals, nbin)
                durations = durations[dropped_intervals==0]
                interval_mask, interval_range_samples = split_interval_duration(durations, interval_range=interval_range, mode=interval_mode, sfreq=sfreq)
                interval_wavgs, interval_sums = compute_fo_stats(itc_sec, divided_intervals, interval_mask, avg_intervals=avg_intervals)
                stats.append({"durations":durations,"intervals": intervals, "interval_wavgs":interval_wavgs, "interval_sums": interval_sums, 
                          "divided_intervals": divided_intervals, "bin_sizes": bin_sizes, "interval_range": interval_range_samples})
        fo_density = collate_stats(stats, 'interval_wavgs', ignore_elements)
        fo_sum = collate_stats(stats, 'interval_sums', ignore_elements)
    return fo_density, fo_sum, stats 



# Example usage
import pickle
alpha = pickle.load(open('/ohba/pi/mwoolrich/mvanes/Projects/Replay/dynamics/nott/hmm/alpha.pkl', 'rb'))
stc = argmax_time_courses(alpha)
fo_density, fo_sum, stats = tinda(stc, nbin=4, interval_range=[0, 0.1, 1, np.inf], interval_mode='sec', sfreq=250, avg_intervals=True)


stc_prim = [1, 0, 0, 1, 0, 1, 0, 0, 1]
stc_sec = [
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
]

nbin = 4

intervals = find_intervals(stc_prim)
divided_intervals = split_intervals(intervals, nbin)
weighted_avgs, interval_sums = compute_weighted_averages(stc_sec, divided_intervals)

print("Intervals:", intervals)
print("Divided intervals:", divided_intervals)
print("Weighted averages:", weighted_avgs)
print("Interval sums:", interval_sums)
