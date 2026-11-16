"""
DyNeMo: Plotting Networks
=========================

In this tutorial we will plot networks from a DyNeMo model trained on source reconstructed MEG data. This tutorial covers:

1. Load regression spectra
2. PSDs
3. Power maps
4. Coherence networks
5. Coherence maps
"""

# sphinx_gallery_thumbnail_number = 7

#%%
# Download post-hoc spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data from `OSF <https://osf.io/by2tc/>`_.

import os

def get_spectra(name, rename):
    os.system(f"osf -p by2tc fetch spectra/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 7 MB)
get_spectra("tde_dynemo_notts_meguk_giles_5_subjects", rename="results/spectra")

#%%
# Load regression spectra
# ^^^^^^^^^^^^^^^^^^^^^^^
# We calculate the networks based on the regression spectra. Let's load these.

import numpy as np

f = np.load("results/spectra/f.npy")
psd = np.load("results/spectra/psd.npy")
coh = np.load("results/spectra/coh.npy")
w = np.load("results/spectra/w.npy")

#%%
# PSDs
# ^^^^
# We can plot the power spectra to see what oscillations typically occur when a particular mode is active. Let's first print the shape of the regression spectra to understand the format of the data.

print(f.shape)
print(psd.shape)

#%%
# We can see: `f` is the frequency axis and `psd` is a (subjects, 2, modes, channels, frequencies) array. The second axis in `psd` corresponds to the regression coefficients and intercept when we calculated the mode spectra using :func:`analysis.spectral.regression_spectra <osl_dynamics.analysis.spectral.regression_spectra>`. The intercept is the time-averaged (static) PSD. The regression coefficients are the change in the PSD from the static PSD for a one standard deviation increase in a mode's mixing coefficient. We plot the regression coefficients. See the :doc:`Spectral Analysis </user_guide/spectra>` page of the User Guide for more details.

psd_coefs = psd[:, 0]
print(psd_coefs.shape)

#%%
# Now let's plot the group-average PSD for each mode.

from osl_dynamics.utils import plotting

# Average over subjects and channels
psd_coefs_mean = np.mean(psd_coefs, axis=(0,2))
print(psd_coefs_mean.shape)

# Plot
n_modes = psd_coefs_mean.shape[0]
fig, ax = plotting.plot_line(
    [f] * n_modes,
    psd_coefs_mean,
    labels=[f"Mode {i}" for i in range(1, n_modes + 1)],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[f[0], f[-1]],
)

#%%
# Power maps
# ^^^^^^^^^^
# Now let's integrate the power spectra to calculate the power.

from osl_dynamics.analysis import power

p = power.variance_from_spectra(f, psd_coefs)
print(p.shape)

#%%
# We can see `p` is a (subjects, modes, channels) array. Let's average over subjects.

mean_p = np.average(p, axis=0, weights=w)
print(mean_p.shape)

#%%
# Now we can plot the power map for each mode. The power maps are the change in power from the static power: red regions have more power when a mode is active and blue regions have less.

# Display the power maps (takes a few seconds to appear)
fig, ax = power.save(
    mean_p,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# We can see recognizable functional networks, which gives us confidence in the DyNeMo fit. We also see the networks are more localized than typical HMM states. A DyNeMo fit often has one mode with a small covariance that takes most of the mixing coefficients: its power map is negative everywhere.
#
# Coherence networks
# ^^^^^^^^^^^^^^^^^^
# Next, let's visualize the coherence networks. First, we need to calculate the networks from the coherence spectra.

print(coh.shape)

#%%
# We can see the coherence spectra is a (subjects, modes, channels, channels, frequencies) array. Let's calculate the mean coherence over all frequencies.

from osl_dynamics.analysis import connectivity

c = connectivity.mean_coherence_from_spectra(f, coh)
print(c.shape)

#%%
# We now have a (subjects, modes, channels, channels) array. This is the change in coherence from the static coherence: positive where the coherence increases when a mode is active and negative where it decreases. Next we average over subjects and show the edges with the largest changes in coherence.

# Average over subjects
mean_c = np.average(c, axis=0, weights=w)

# Keep the 3% of edges with the largest changes (increases or decreases)
thres_mean_c = connectivity.threshold(mean_c, percentile=97, absolute_value=True)

#%%
# Now we can visualize the networks.

connectivity.save(
    thres_mean_c,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# We can see the increases in coherence (red) are in the same regions as the increases in power. We expect these networks will improve with more subjects.
#
# Coherence maps
# ^^^^^^^^^^^^^^
# We can display the coherence as a spatial map rather than a graphical network by averaging the edges for each parcel (excluding the diagonal, which is the relative change in power).

n_parcels = mean_c.shape[-1]
mean_c[:, range(n_parcels), range(n_parcels)] = 0
mean_c_map = np.sum(mean_c, axis=-1) / (n_parcels - 1)

fig, ax = power.save(
    mean_c_map,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)
