"""Example code for plotting power maps calculated using the state/mode spectra
estimated from the source reconstructed data and inferred state/mode time course.

The spectra can be calculate with a multitaper (in the case of a state time
course) or regression (in the case of a mode time course).

See examples/minimal/multitaper_spectra.py for how to calculate a multitaper
and examples/minimal/regression_spectra.py for how to calculate regression spectra.

In this script we assume this has been done and we have the subject-specific spectra
files:
- f.npy, the frequency axis.
- psd.npy, the subject-specific power spectra.
- w.npy, the weight for each subject (used for calculating the group average).

The other file: coh.npy is not needed for this script.
"""

import numpy as np

from osl_dynamics.analysis import power

# Load the subject-level spectra
f = np.load("f.npy")
psd = np.load("psd.npy")
w = np.load("w.npy")

# Calculate the group average
gpsd = np.average(psd, axis=0, weights=w)

# Calculate power maps from the spectra
# (frequency_range is an optional argument)
power_map = power.variance_from_spectra(f, gpsd, frequency_range=[1, 30])

# Save the power maps as images
power.save(
    power_map,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
    filename="maps_.png",
)
