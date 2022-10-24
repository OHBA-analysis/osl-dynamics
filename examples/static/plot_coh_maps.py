"""Example script for plotting coherence maps from static spectra.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics.analysis import connectivity

# Make directory to hold plots
os.makedirs("figures", exist_ok=True)

# Load spectra (calculated with static/calc_spectra.py)
f = np.load("spectra/f.npy")
coh = np.load("spectra/coh.npy")

# Plot subject specific coherence maps
# Here, we just look at alpha activity (i.e. the 10 Hz activity)
conn_map = connectivity.mean_coherence_from_spectra(f, coh, frequency_range=[8, 12])
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/coh_subj_.png",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    threshold=0.95,
)

# Plot the group mean
conn_map = np.mean(conn_map, axis=0)
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/coh_group_.png",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    threshold=0.95,
)
