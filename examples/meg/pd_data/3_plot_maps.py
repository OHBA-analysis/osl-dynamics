"""Plot power maps and coherence networks for each state.

"""

import os
import numpy as np

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

# Directories
plots_dir = "plots"
spectra_dir = "results/spectra"

os.makedirs(plots_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"

#%% Load spectra

f = np.load(spectra_dir + "/f.npy")  # (n_freq,)
psd = np.load(spectra_dir + "/psd.npy")  # (n_subjects, n_states, n_parcels, n_freq)
coh = np.load(spectra_dir + "/coh.npy")  # (n_subjects, n_states, n_parcels, n_parcels, n_freq)
w = np.load(spectra_dir + "/w.npy")  # (n_subjects,)

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
        filename=plots_dir + f"/psd_{i}.png",
    )

#%% Plot power maps

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Calculate the power map by integrating the power spectra over a frequency range
p = power.variance_from_spectra(f, gpsd, frequency_range=[2, 30])

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    filename=plots_dir + "/pow_.png",
    subtract_mean=True,
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": 1, "darkness": 0.4, "alpha": 1},
)

#%% Plot coherence networks

# Calculate the group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate the coherence network by averaging over a frequency range
c = connectivity.mean_coherence_from_spectra(f, gcoh, frequency_range=[2, 20])

# Threshold the top 2% of connections
c = connectivity.threshold(c, percentile=98, subtract_mean=True)

# Plot
connectivity.save(
    c,
    filename=plots_dir + "/coh_.png",
    parcellation_file=parcellation_file,
    plot_kwargs={
        "edge_cmap": "red_transparent_full_alpha_range",
        "display_mode": "lyrz",
    },
)
