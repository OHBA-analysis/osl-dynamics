"""Plot TDE-HMM networks and summary statistics.

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from osl_dynamics import array_ops
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

# Set global matplotlib style
plotting.set_style(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 28,
    }
)

# Directories
inf_params_dir = "ae_results/inf_params"
plots_dir = "ae_results/plots"

os.makedirs(plots_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"

#%% Load the inferred parameters

# State probabilities
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

# State means and covariances
means = np.load(inf_params_dir + "/means.npy")
covs = np.load(inf_params_dir + "/covs.npy")

#%% Plot mean activity maps

# Plot
power.save(
    means,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    plot_kwargs={"views": ["lateral"]},
    filename=plots_dir + "/mean_.png",
)

#%% Plot functional connectivity networks

# Convert the state covariances into correlation matrices
corrs = abs(array_ops.cov2corr(covs))

# Plot
connectivity.save(
    corrs,
    parcellation_file=parcellation_file,
    threshold=0.97,  # plot the top 3% of connections
    plot_kwargs={"edge_cmap": "Reds"},
    filename=plots_dir + "/corr_.png",
)

#%% Calculate summary statistics

# Convert to a state time course by taking the most probable state at
# each time point (Viterbi path)
stc = modes.argmax_time_courses(alp)

# Calculate fractional occupancy
fo = modes.fractional_occupancies(stc)  # (n_subjects, n_states)

# Calculate mean lifetimes (s)
lt = modes.mean_lifetimes(stc, sampling_frequency=250)  # (n_subjects, n_states)

# Calculate mean interval times (s)
intv = modes.mean_intervals(stc, sampling_frequency=250)  # (n_subjects, n_states)

# Calculate switching rates (/s)
sr = modes.switching_rates(stc, sampling_frequency=250)  # (n_subjects, n_states)

#%% Plot summary statistics

def save(filename):
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()

# Create a pandas DataFrame containing the data
# This will help with plotting
ss_dict = {
    "Fractional Occupancy": [],
    "Mean Lifetime (s)": [],
    "Mean Interval (s)": [],
    "Switching Rate (/s)": [],
    "State": [],
}
n_subjects, n_states = fo.shape
for subject in range(n_subjects):
    for state in range(n_states):
        ss_dict["Fractional Occupancy"].append(fo[subject, state])
        ss_dict["Mean Lifetime (s)"].append(lt[subject, state])
        ss_dict["Mean Interval (s)"].append(intv[subject, state])
        ss_dict["Switching Rate (/s)"].append(sr[subject, state])
        ss_dict["State"].append(state + 1)
ss_df = pd.DataFrame(ss_dict)

# Close all previous figures
plt.close("all")

# Plot fractional occupancies
sns.violinplot(data=ss_df, x="State", y="Fractional Occupancy")
save(plots_dir + "/sum_stats_fo.png")

# Plot mean lifetimes
sns.violinplot(data=ss_df, x="State", y="Mean Lifetime (s)")
save(plots_dir + "/sum_stats_lt.png")

# Plot mean intervals
sns.violinplot(data=ss_df, x="State", y="Mean Interval (s)")
save(plots_dir + "/sum_stats_intv.png")

# Plot switching rates
sns.violinplot(data=ss_df, x="State", y="Switching Rate (/s)")
save(plots_dir + "/sum_stats_sr.png")
