"""Example script for plotting power maps from static spectra.

"""

print("Setting up")
import os
from osl_dynamics.analysis import power, spectral

# Make directory to hold plots
os.makedirs("figures", exist_ok=True)

# Load spectra (calculated with static/calc_spectra.py)
spectra = spectral.Spectra.load("spectra/spectra.npz")

# Source reconstruction files used to create the source space data
parcellation_file = (
    "atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz"
)

# Subject specific power maps
power_map = spectra.power_maps()
power.save(
    power_map=power_map,
    filename="figures/power_subj_.png",
    parcellation_file=parcellation_file,
    subtract_mean=True,  # display the differences in power relative to the group mean
)

# Group level power map
power_map = spectra.group_average().power_maps()
power.save(
    power_map=power_map,
    filename="figures/power_group_.png",
    parcellation_file=parcellation_file,
)
