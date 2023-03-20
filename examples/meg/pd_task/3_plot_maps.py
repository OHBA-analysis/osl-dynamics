"""Plot power maps and coherence networks for each state.

"""

import os
import numpy as np

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

# Directories
plots_dir = "results/plots"
spectra_dir = "results/spectra"

os.makedirs(plots_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

#%% Load spectra

f = np.load(spectra_dir + "/f.npy")  # (n_freq,)
psd = np.load(spectra_dir + "/psd.npy")  # (n_subjects, n_states, n_parcels, n_freq)
coh = np.load(spectra_dir + "/coh.npy")  # (n_subjects, n_states, n_parcels, n_parcels, n_freq)
w = np.load(spectra_dir + "/w.npy")  # (n_subjects,)
wb_comp = np.load(spectra_dir + "/wb_comp.npy")  # (n_components, n_freq)

# Plot NNMF
plotting.plot_line(
    [f] * wb_comp.shape[0],
    wb_comp,
    x_label="Frequency (Hz)",
    y_label="Weighting",
    filename=plots_dir + "/wb_comp.png",
)

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
        errors=[[p-e], [p+e]],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=plots_dir + f"/psd_{i}.png",
    )

#%% Plot power maps

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Calculate the power map by integrating the power spectra over a frequency range
p = power.variance_from_spectra(f, gpsd, wb_comp)

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    component=0,
    subtract_mean=True,
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": 1, "darkness": 0.4, "alpha": 1},
    filename=plots_dir + "/pow_.png",
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
    plot_kwargs={"edge_cmap": "Reds"},
    filename=plots_dir + "/coh_.png",
)
