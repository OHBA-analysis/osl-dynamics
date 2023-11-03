"""Example code for plotting connectivity maps using the coherence
from state/mode spectra.

The spectra can be calculate with a multitaper (in the case of a state time
course) or regression (in the case of a mode time course).

See examples/minimal/multitaper_spectra.py for how to calculate a multitaper
and examples/minimal/regression_spectra.py for how to calculate regression spectra.

In this script we assume this has been done and we have the group-average spectra
files: f.npy and coh.npy. (The other file: psd.npy is not needed for this script.)
"""

import numpy as np

from osl_dynamics.analysis import connectivity

# Load subject-specific state/mode spectra
f = np.load("f.npy")
coh = np.load("coh.npy")
w = np.load("w.npy")

# Calculate group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate connectivity maps from coherence spectra
# The frequency_range argument is optional. It can be replaced with a
# spectral component calculate with NNMF, see connectivity-maps_spectra-nnmf.py
conn_map = connectivity.mean_coherence_from_spectra(f, gcoh, frequency_range=[1, 25])

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will plot the top 5% of connections above/below the mean connectivity
# across states/modes
conn_map = connectivity.threshold(conn_map, percentile=95, subtract_mean=True)

# Plot connectivity maps
connectivity.save(
    conn_map,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    filename="coh_.png",
)
