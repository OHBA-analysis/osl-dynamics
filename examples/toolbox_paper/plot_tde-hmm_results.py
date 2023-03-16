"""Plot TDE-HMM networks and summary statistics.

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from osl_dynamics.analysis import spectral, power, connectivity
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
inf_params_dir = "tde_results/inf_params"
spectra_dir = "tde_results/spectra"
plots_dir = "tde_results/plots"

os.makedirs(plots_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"

#%% Load spectra

f = np.load(spectra_dir + "/f.npy")  # (n_freq,)
psd = np.load(spectra_dir + "/psd.npy")  # (n_subjects, n_states, n_parcels, n_freq)
coh = np.load(spectra_dir + "/coh.npy")  # (n_subjects, n_states, n_parcels, n_parcels, n_freq)

# Rescale the PSDs so we have more legible ticks/colourbars
# (The PSDs have arbitrary units)
psd *= 100

#%% Calculate non-negative matrix factorization (NNMF)

# We perform NNMF on the stacked coherence matrices from each subject
# By fitting 2 components we will find 2 frequency bands for coherent activity
nnmf = spectral.decompose_spectra(coh, n_components=2)

# Plot
plotting.plot_line(
    [f] * nnmf.shape[0],
    nnmf,
    x_label="Frequency (Hz)",
    y_label="Weighting",
    filename=plots_dir + "/nnmf.png",
)

#%% Plot power spectra

# Calculate the group average power spectrum for each state
gpsd = np.mean(psd, axis=0)

# Plot
for i in range(gpsd.shape[0]):
    p = np.mean(gpsd[i], axis=0)  # mean over parcels
    e = np.std(gpsd[i]) / np.sqrt(gpsd[i].shape[0])  # standard error on the mean
    plotting.plot_line(
        [f],
        [p],
        errors=[[p - e], [p + e]],
        labels=[f"State {i + 1}"],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        plot_kwargs={"linewidth": 2},
        filename=plots_dir + f"/psd_{i}.png",
    )

#%% Plot power maps

# Calculate the group average power spectrum for each state
gpsd = np.mean(psd, axis=0)

# Calculate the power map by integrating over each component from the NNMF
p = power.variance_from_spectra(f, gpsd, nnmf)

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    component=0,
    subtract_mean=True,
    plot_kwargs={"views": ["lateral"]},
    filename=plots_dir + "/pow_.png",
)

#%% Plot coherence networks

# Calculate the group average
gcoh = np.mean(coh, axis=0)

# Calculate the coherence network by averaging over each component from the NNMF
c = connectivity.mean_coherence_from_spectra(f, gcoh, nnmf)

# Use a GMM to theshold which edges to show
c = connectivity.gmm_threshold(c, standardize=True, p_value=0.01, subtract_mean=True)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    component=0,
    plot_kwargs={"edge_cmap": "Reds"},
    filename=plots_dir + "/coh_.png",
)

#%% Calculate summary statistics

# Load state probabilities
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

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
