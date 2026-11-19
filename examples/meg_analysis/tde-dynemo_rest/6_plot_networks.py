"""Plot group-average power maps and coherence networks for each mode.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of modes and run id, e.g. python 6_plot_networks.py 6 1")
    exit()
n_modes = int(argv[1])
run = int(argv[2])

#%% Import packages

import os
from osl_dynamics.analysis import spectral, power, connectivity

#%% Setup directories and files

spectra_dir = f"results/{n_modes}_modes/run{run:02d}/spectra"
networks_dir = f"results/{n_modes}_modes/run{run:02d}/networks"

os.makedirs(networks_dir, exist_ok=True)

parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

#%% Load spectra

spectra = spectral.Spectra.load(f"{spectra_dir}/spectra.npz")

# Group average
group = spectra.group_average()

#%% Plot power spectra

# Regression coefficients for each mode (averaged over parcels)
for i in range(group.n_states):
    group.plot_psd(states=[i], filename=f"{networks_dir}/psd_{i}.png")

#%% Plot power maps

# Calculate the power map by averaging the coefficients over all frequencies
p = group.power_maps()  # (n_modes, n_parcels)

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    plot_kwargs={"symmetric_cbar": True},
    filename=f"{networks_dir}/pow_.png",
)

#%% Plot coherence networks

# Keep the 3% of edges with the largest changes (increases or decreases)
c = group.coherence_networks(percentile=97)  # (n_modes, n_parcels, n_parcels)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    filename=f"{networks_dir}/coh_.png",
)
