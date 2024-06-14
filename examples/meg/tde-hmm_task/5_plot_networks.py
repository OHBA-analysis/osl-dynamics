"""Plot group-average power maps and coherence networks for each state.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 5_plot_networks.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import numpy as np

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

#%% Setup directories and files

# Directories
spectra_dir = f"results/{n_states}_states/run{run:02d}/spectra"
networks_dir = f"results/{n_states}_states/run{run:02d}/networks"

os.makedirs(networks_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

#%% Load spectra

f = np.load(f"{spectra_dir}/f.npy")  # (n_freq,)
psd = np.load(f"{spectra_dir}/psd.npy")  # (n_subjects, n_states, n_parcels, n_freq)
coh = np.load(f"{spectra_dir}/coh.npy")  # (n_subjects, n_states, n_parcels, n_parcels, n_freq)
w = np.load(f"{spectra_dir}/w.npy")  # (n_subjects,)

wb_comp = np.load(f"{spectra_dir}/nnmf_2.npy")  # (n_components, n_freq)

#%% Plot power spectra

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Plot
for i in range(gpsd.shape[0]):
    p = np.mean(gpsd[i], axis=0)  # mean over parcels
    e = np.std(gpsd[i]) / np.sqrt(gpsd[i].shape[0])  # standard error on the mean
    plotting.plot_line(
        [f],
        [p],
        errors=[[p - e], [p + e]],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=f"{networks_dir}/psd_{i:02d}.png",
    )

#%% Plot power maps

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Calculate the power map by integrating the power spectra over a frequency range
p = power.variance_from_spectra(f, gpsd, wb_comp)

# Plot
power.save(
    p,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    component=0,
    plot_kwargs={"symmetric_cbar": True},
    filename=f"{networks_dir}/pow_.png",
)

#%% Plot coherence networks

# Calculate the group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate the coherence network by averaging over a frequency range
c = connectivity.mean_coherence_from_spectra(f, gcoh, wb_comp)

# Threshold the top 2% of connections
c = connectivity.threshold(c, percentile=98, subtract_mean=True)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    component=0,
    filename=f"{networks_dir}/coh_.png",
)
