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






import numpy as np

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


def divide_intervals(intervals, nbin=2):
    divided_intervals = []
    bin_sizes = []
    for start, end in intervals:
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
    return divided_intervals, bin_sizes


def compute_weighted_averages(tc_sec, divided_intervals):
    interval_wavgs = []
    interval_sums = []
    for row in tc_sec.T:
        fo_row = np.mean(row)
        weighted_avg = np.zeros((len(divided_intervals), len(divided_intervals[0])))
        interval_sum = np.zeros((len(divided_intervals), len(divided_intervals[0])))
        for i, interval in enumerate(divided_intervals):
            for j, (start, end) in enumerate(interval):
                bin_sum = np.sum(row[start:end])
                bin_avg = bin_sum / (end - start)
                weighted_avg[i,j] = bin_avg * fo_row
                interval_sum[i,j] = bin_sum
        interval_wavgs.append(weighted_avg)
        interval_sums.append(interval_sum)
    return interval_wavgs, interval_sums



alpha = pickle.load(open('/ohba/pi/mwoolrich/mvanes/Projects/Replay/dynamics/nott/hmm/alpha.pkl', 'rb'))
stc = argmax_time_courses(alpha)

def compute_fo_density(tc, nbin=2, interval_range=None, interval_mode=None):
    if isinstance(tc, list): # list of time courses (e.g., subjects' HMM state time courses)
        fo_density = [zip(compute_fo_density(itc, nbin, interval_range, interval_mode)) for itc in tc]
    else:
        fo_density = []
        dim = tc.shape
        for i in range(dim[1]):
            itc_prim = tc[:, i]
            if not np.array_equal(itc_prim, itc_prim.astype(int)):
                fo_density.append(None)
            else:
                itc_sec = tc[:, np.setdiff1d(range(dim[1]), i)]
                intervals, durations = find_intervals(itc_prim)
                nbins = 4
                divided_intervals, bin_sizes = divide_intervals(intervals, nbins)
                interval_wavgs, interval_sums = compute_weighted_averages(itc_sec, divided_intervals)
            fo_density.append({"interval_wavgs":interval_wavgs, "interval_sums": interval_sums,
                              "divided_intervals": divided_intervals, "bin_sizes": bin_sizes,
                                "intervals": intervals, "durations": durations})
    return fo_density # TODO: might have to choose to average either over intervals or over subjects

# Example usage
stc_prim = [1, 0, 0, 1, 0, 1, 0, 0, 1]
stc_sec = [
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
]

nbin = 4

intervals = find_intervals(stc_prim)
divided_intervals = divide_intervals(intervals, nbin)
weighted_avgs, interval_sums = compute_weighted_averages(stc_sec, divided_intervals)

print("Intervals:", intervals)
print("Divided intervals:", divided_intervals)
print("Weighted averages:", weighted_avgs)
print("Interval sums:", interval_sums)
