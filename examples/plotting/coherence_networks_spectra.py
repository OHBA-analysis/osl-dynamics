"""Example code for plotting connectivity maps using the coherence
from state/mode spectra.

The spectra can be calculate with a multitaper (in the case of a state time
course) or regression (in the case of a mode time course).

See examples/spectra/hmm_multitaper.py for how to calculate a multitaper
and examples/spectra/dynemo_regression_spectra.py for how to calculate regression spectra.

In this script we assume this has been done and the subject-specific spectra
have been saved with spectra.save("spectra.npz").
"""

from osl_dynamics.analysis import connectivity, spectral

# Load subject-specific state/mode spectra
spectra = spectral.Spectra.load("spectra.npz")

# Calculate group average
group = spectra.group_average()

# Calculate connectivity maps from the change in coherence from the static
# spectrum and keep the 3% of connections with the largest changes
# (increases or decreases)
# The frequency_range argument is optional. It can be replaced with
# spectral components calculated with NNMF, see coherence_networks_spectra_nnmf.py
conn_map = group.coherence_networks(frequency_range=[1, 25], percentile=97)

# Plot connectivity maps
connectivity.save(
    conn_map,
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
    filename="coh_.png",
)
