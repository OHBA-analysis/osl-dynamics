"""Example code for plotting connectivity maps using the coherence
from state spectra and spectral components calculated with NNMF.

The spectra are calculated with a multitaper, see
examples/spectra/hmm_multitaper.py, and saved with spectra.save("spectra.npz").
"""

from osl_dynamics.analysis import connectivity, spectral
from osl_dynamics.utils import plotting

# Load subject-specific state spectra
spectra = spectral.Spectra.load("spectra.npz")

# Calculate spectral components using subject-specific coherences
wb_comp = spectra.decompose(n_components=2)

# Plot the spectral components
f = spectra.frequencies
plotting.plot_line([f, f], wb_comp, filename="wideband.png")

# Group average
group = spectra.group_average()

# Calculate connectivity maps for each spectral component from the change
# in coherence from the static spectrum and keep the 3% of connections with
# the largest changes (increases or decreases)
conn_map = group.coherence_networks(bands=wb_comp, percentile=97)

# Plot connectivity maps
connectivity.save(
    conn_map,
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
    component=0,  # only plot the first spectral component
    filename="coh_.png",
)
