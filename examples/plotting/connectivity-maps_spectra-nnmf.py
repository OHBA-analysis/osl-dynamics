"""Example code for plotting connectivity maps using the coherence
from state/mode spectra.

The spectra can be calculate with a multitaper (in the case of a state time
course) or regression (in the case of a mode time course).

See examples/analysis/multitaper_spectra.py for how to calculate a multitaper
and examples/analysis/regression_spectra.py for how to calculate a regression.

NOTE: this script requires the subject-specific spectra. To get this we need
to modify the multitaper_spectra.py/regression_spectra.py script to save
the subject specific spectra instead of the group average. We do this by
commenting the np.average() line and optionally saving the weights "w" (as
w.npy).
"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import connectivity, spectral
from osl_dynamics.data import OSL_HMM
from osl_dynamics.utils import plotting

# Source reconstruction files used to create the training data
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Load subject-specific state/mode spectra
coh = np.load("coh.npy")
w = np.load("w.npy")

# Group-level coherence matrix
gcoh = np.average(coh, axis=0, weights=w)

# Calculate spectral components using subject-specific coherences
wideband_components = spectral.decompose_spectra(coh, n_components=2)

# Plot the spectral components
plotting.plot_line([f, f], wideband_components, filename="wideband.png")

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will plot the top X % of connections above/below the mean connectivity
# across states/modes where we use a GMM to calculate the threshold X
conn_map = connectivity.gmm_threshold(
    conn_map,
    subtract_mean=True,
    standardize=True,
    one_component_percentile=95,
    filename="gmm_conn_.png",
    plot_kwargs={
        "x_label": "Standardised Relative Coherence",
        "y_label": "Probability",
    },
)

# Plot connectivity maps
connectivity.save(
    connectivity_map=conn_map,
    filename="coh_.png",
    parcellation_file=parcellation_file,
    component=0,  # only plot the first spectral component
)
