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
import pickle

# def validate_data(stc):
#     if isinstance(stc, list):
#         if not np.array_equal(stc[0], stc[0].astype(int)):
#         stc = np.array(stc)
#     if len(stc.shape) == 1:
#         stc = np.expand_dims(stc, axis=0)
#     return stc

def find_intervals(tc_hot):
    intervals = []
    durations = []
    tc_tmp = np.insert(np.insert(tc_hot, 0, 0, axis=0), -1, 0, axis=0)
    start = np.where(np.diff(tc_tmp)==1)[0]
    end = np.where(np.diff(tc_tmp)==-1)[0]
    intervals = list(zip(end[:-1], start[1:]))
    durations = np.diff(intervals, axis=1).squeeze()
    return intervals, durations


def split_intervals(intervals, nbin=2):
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
    if mode == "sec" and sfreq is None:
        raise ValueError("Sampling frequency (sfreq) must be specified when mode is 'sec'")
    if interval_range is None:
        return [np.ones_like(durations)]
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



def compute_weighted_averages(tc_sec, divided_intervals, interval_mask=None, avg_intervals=True):
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


alpha = pickle.load(open('/ohba/pi/mwoolrich/mvanes/Projects/Replay/dynamics/nott/hmm/alpha.pkl', 'rb'))
stc = argmax_time_courses(alpha)

def tinda(tc, nbin=2, interval_range=None, interval_mode=None, sfreq=None, avg_intervals=True):
    if isinstance(tc, list): # list of time courses (e.g., subjects' HMM state time courses)
        fo_density, fo_sum, stats = zip(*[tinda(itc, nbin, interval_range, interval_mode, sfreq, avg_intervals) for itc in tc])
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
                itc_sec = tc[:, np.setdiff1d(range(dim[1]), i)]
                intervals, durations = find_intervals(itc_prim)
                divided_intervals, bin_sizes, dropped_intervals = split_intervals(intervals, nbin)
                durations = durations[dropped_intervals==0]
                interval_mask, interval_range_samples = split_interval_duration(durations, interval_range=interval_range, mode=interval_mode, sfreq=sfreq)
                interval_wavgs, interval_sums = compute_weighted_averages(itc_sec, divided_intervals, interval_mask, avg_intervals=avg_intervals)
                stats.append({"durations":durations,"intervals": intervals, "interval_wavgs":interval_wavgs, "interval_sums": interval_sums, 
                          "divided_intervals": divided_intervals, "bin_sizes": bin_sizes, "interval_range": interval_range_samples})
        fo_density = collate_stats(stats, 'interval_wavgs', ignore_elements)
        fo_sum = collate_stats(stats, 'interval_sums', ignore_elements)
    return fo_density, fo_sum, stats 


def collate_stats(stats, field, ignore_elements=None):
    num_states = len(stats)
    nbins = stats[0][field][0].shape[-1]
    n_ranges = len(stats[0][field])
    collated_stat = [np.full((num_states, num_states, nbins), np.nan) for _ in range(n_ranges)]
    for i in range(num_states):
        if i in ignore_elements:
            continue
        for k in range(n_ranges):
            collated_stat[k][i, np.setdiff1d(range(num_states),i)] = stats[i][field][k]
    return collated_stat

# Example usage
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
