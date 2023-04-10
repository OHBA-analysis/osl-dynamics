"""Calculate summary statistics for each state and do max-stat permutation
testing for significance.

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from osl_dynamics.analysis import statistics
from osl_dynamics.inference import modes

# Directories
inf_params_dir = "results/inf_params"
summary_stats_dir = "results/summary_stats"

os.makedirs(summary_stats_dir, exist_ok=True)

#%% Load state time course

# State probabilities
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

# Calculate a state time course by taking the most likely state
stc = modes.argmax_time_courses(alp)

#%% Calculate summary statistics

# Fractional occupancy
fo = modes.fractional_occupancies(stc)

# Mean lifetime
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

# Mean interval
intv = modes.mean_intervals(stc, sampling_frequency=250)

# Mean switching rate
sr = modes.switching_rates(stc, sampling_frequency=250)

np.save(summary_stats_dir + "/fo.npy", fo)
np.save(summary_stats_dir + "/lt.npy", lt)
np.save(summary_stats_dir + "/intv.npy", intv)
np.save(summary_stats_dir + "/sr.npy", sr)

#%% Max-stat permutation testing

# Load confounds
behav_data = pd.read_csv("data/BehaveData_SourceAnalysis.csv")

# Group assignments
assignments = behav_data["Group"].values

# Confounds
covariates = {
    "age": behav_data["Age"].values,
    "gender": behav_data["Gender"].values,
    "handedness": behav_data["Handedness"].values,
    "education": behav_data["Education"].values,
}

# Fit a GLM for each summary statistic
pvalues = []
for metric in ["fo", "lt", "intv", "sr"]:
    data = np.load(f"{summary_stats_dir}/{metric}.npy")
    _, p = statistics.group_diff_max_stat_perm(
        data=data,
        assignments=assignments,
        n_perm=10000,
        covariates=covariates,
        n_jobs=4,
    )
    pvalues.append(p)

#%% Plot


def save(filename):
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()


def print_significance(p):
    sig = []
    for p_ in p:
        if p_ < 0.005:
            sig.append("**")
        elif p_ < 0.025:
            sig.append("*")
        else:
            sig.append("-")
    print(" ".join(sig))


# Healthy control or PD patient
behav_data = pd.read_csv("../data/behav_data/BehaveData_SourceAnalysis.csv")
assignments = ["HC" if a == 1 else "PD" for a in behav_data["Group"].values]

# Load summary stats
fo = np.load(summary_stats_dir + "/fo.npy")
lt = np.load(summary_stats_dir + "/lt.npy")
intv = np.load(summary_stats_dir + "/intv.npy")
sr = np.load(summary_stats_dir + "/sr.npy")

# Create a pandas DataFrame containing the data
# This will help with plotting
ss_dict = {
    "Fractional Occupancy": [],
    "Mean Lifetime (s)": [],
    "Mean Interval (s)": [],
    "Switching Rate (/s)": [],
    "State": [],
    "Group": [],
}
n_subjects, n_states = fo.shape
for subject in range(n_subjects):
    for state in range(n_states):
        ss_dict["Fractional Occupancy"].append(fo[subject, state])
        ss_dict["Mean Lifetime (s)"].append(lt[subject, state])
        ss_dict["Mean Interval (s)"].append(intv[subject, state])
        ss_dict["Switching Rate (/s)"].append(sr[subject, state])
        ss_dict["State"].append(state + 1)
        ss_dict["Group"].append(assignments[subject])
ss_df = pd.DataFrame(ss_dict)

# Plot frational occupancies
sns.violinplot(data=ss_df, x="State", y="Fractional Occupancy", hue="Group", split=True)
save("plots/fo.png")
print_significance(pvalues[0])

# Plot mean lifetimes
sns.violinplot(data=ss_df, x="State", y="Mean Lifetime (s)", hue="Group", split=True)
save("plots/lt.png")
print_significance(pvalues[1])

# Plot mean intervals
sns.violinplot(data=ss_df, x="State", y="Mean Interval (s)", hue="Group", split=True)
save("plots/intv.png")
print_significance(pvalues[2])

# Plot switching rates
sns.violinplot(data=ss_df, x="State", y="Switching Rate (/s)", hue="Group", split=True)
save("plots/sr.png")
print_significance(pvalues[3])
