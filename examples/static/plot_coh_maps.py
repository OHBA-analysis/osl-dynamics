"""Example script for plotting coherence maps from static spectra.

"""

print("Setting up")
import os
from osl_dynamics.analysis import connectivity, spectral

# Make directory to hold plots
os.makedirs("figures", exist_ok=True)

# Load spectra (calculated with static/calc_spectra.py)
spectra = spectral.Spectra.load("spectra/static.npz")

# Plot subject specific coherence maps
# Here, we just look at alpha activity (i.e. the 10 Hz activity)
# and keep the 3% of edges with the largest coherence
conn_map = spectra.coherence_networks(frequency_range=[8, 13], percentile=97)
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/coh_subj_.png",
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
)

# Plot the group average
conn_map = spectra.group_average().coherence_networks(
    frequency_range=[8, 13], percentile=97
)
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/coh_group_.png",
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
)
