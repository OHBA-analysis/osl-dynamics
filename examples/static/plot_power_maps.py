"""Example script for plotting power maps from static spectra.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics.analysis import power

# Make directory to hold plots
os.makedirs("figures", exist_ok=True)

# Load spectra (calculated with static/calc_spectra.py)
f = np.load("spectra/f.npy")
psd = np.load("spectra/psd.npy")

# Source reconstruction files used to create the source space data
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Subject specific power maps
power_map = power.variance_from_spectra(f, psd)
power.save(
    power_map=power_map,
    filename="figures/power_subj_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,  # display the differences in power relative to the group mean
)

# Group level power map
power_map = np.mean(power_map, axis=0)
power.save(
    power_map=power_map,
    filename="figures/power_group_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
)
