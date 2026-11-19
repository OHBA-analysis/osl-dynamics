"""Plot group-average power maps and coherence networks for each state.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 6_plot_networks.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

import os
import numpy as np
from osl_dynamics.analysis import spectral, power, connectivity

#%% Setup directories and files

spectra_dir = f"results/{n_states:02d}_states/run{run:02d}/spectra"
networks_dir = f"results/{n_states:02d}_states/run{run:02d}/networks"

os.makedirs(networks_dir, exist_ok=True)

parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

#%% Load spectra

spectra = spectral.Spectra.load(f"{spectra_dir}/spectra.npz")
wb_comp = np.load(f"{spectra_dir}/nnmf_2.npy")  # (n_components, n_freq)

# Group average
group = spectra.group_average()

#%% Plot power spectra

# PSD of each state (averaged over parcels) with the static PSD
for i in range(group.n_states):
    group.plot_psd(
        values="psd",
        states=[i],
        static=True,
        filename=f"{networks_dir}/psd_{i:02d}.png",
    )

#%% Plot power maps

# Change in power from the static power for each nnmf component
p = group.power_maps(bands=wb_comp)  # (n_components, n_states, n_parcels)

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    component=0,  # nnmf component
    plot_kwargs={"symmetric_cbar": True},
    filename=f"{networks_dir}/pow_.png",
)

#%% Plot coherence networks

# Change in coherence from the static coherence, keeping the top 3% of edges
c = group.coherence_networks(bands=wb_comp, percentile=97)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    component=0,  # nnmf component
    filename=f"{networks_dir}/coh_.png",
)
