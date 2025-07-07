"""Plot group-average power maps and coherence networks for each mode.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of modes and run id, e.g. python 6_plot_networks.py 6 1")
    exit()
n_modes = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")
import os
import numpy as np
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

#%% Setup directories and files

spectra_dir = f"results/{n_modes}_modes/run{run:02d}/spectra"
networks_dir = f"results/{n_modes}_modes/run{run:02d}/networks"

os.makedirs(networks_dir, exist_ok=True)

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

#%% Load spectra

f = np.load(f"{spectra_dir}/f.npy")  # (n_freq,)
psd = np.load(f"{spectra_dir}/psd.npy")  # (n_subjects, 2, n_modes, n_parcels, n_freq)
coh = np.load(f"{spectra_dir}/coh.npy")  # (n_subjects, n_modes, n_parcels, n_parcels, n_freq)
w = np.load(f"{spectra_dir}/w.npy")  # (n_subjects,)

#%% Plot power spectra

gpsd = np.average(psd, axis=0, weights=w)
gpsd_coefs = gpsd[0]
for i in range(gpsd_coefs.shape[0]):
    p = np.mean(gpsd_coefs[i], axis=0)  # mean over parcels
    plotting.plot_line(
        [f],
        [p],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=f"{networks_dir}/psd_{i}.png",
    )

#%% Plot power maps

# Calculate the power map by integrating the power spectra over all frequencies
p = power.variance_from_spectra(f, gpsd_coefs)

# Plot
power.save(
    p,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,  # just for visualisation
    filename=f"{networks_dir}/pow_.png",
)

#%% Plot coherence networks

# Calculate the group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate the coherence network by averaging over all frequencies
c = connectivity.mean_coherence_from_spectra(f, gcoh)

# Threshold the top 2% of connections
c = connectivity.threshold(c, percentile=98, subtract_mean=True)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    filename=f"{networks_dir}/coh_.png",
)
