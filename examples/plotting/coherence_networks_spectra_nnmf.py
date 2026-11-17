"""Example code for plotting connectivity maps using the coherence
from state spectra and spectral components calculated with NNMF.

The spectra are calculated with a multitaper, see
examples/spectra/hmm_multitaper.py. NNMF needs non-negative coherences, so it
cannot be used with the coherence from regression spectra, which can be
negative.
"""

import numpy as np

from osl_dynamics.analysis import connectivity, spectral
from osl_dynamics.utils import plotting

# Load subject-specific state spectra
f = np.load("f.npy")
coh = np.load("coh.npy")
w = np.load("w.npy")

# Calculate spectral components using subject-specific coherences
wb_comp = spectral.decompose_spectra(coh, n_components=2)

# Plot the spectral components
plotting.plot_line([f, f], wb_comp, filename="wideband.png")

# Group-level coherences
gcoh = np.average(coh, axis=0, weights=w)

# Calculate connectivity maps from coherence spectra for each spectral component
conn_map = connectivity.mean_coherence_from_spectra(f, gcoh, wb_comp)

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
        "x_label": "Standardized Relative Coherence",
        "y_label": "Probability",
    },
)

# Plot connectivity maps
connectivity.save(
    conn_map,
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
    component=0,  # only plot the first spectral component
    filename="coh_.png",
)
