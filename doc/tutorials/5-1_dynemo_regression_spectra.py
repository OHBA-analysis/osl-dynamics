"""
DyNeMo: Regression Spectra
==========================

In this tutorial we calculate subject and mode-specific spectra using a trained DyNeMo model and **source space** data. Here, we should **not** use the prepared (e.g. TDE-PCA) data.
"""

# sphinx_gallery_thumbnail_path = "_static/tutorial_thumbs/5-1_dynemo_regression_spectra.png"

#%%
# Download source data
# ^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data from `OSF <https://osf.io/by2tc/>`_.

import os

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 52 MB)
get_data("notts_meguk_giles_5_subjects", rename="source_data")

#%%
# Download inferred parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_inf_params(name, rename):
    os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 11 MB)
get_inf_params("tde_dynemo_notts_meguk_giles_5_subjects", rename="results/inf_params")

#%%
# Load the source data
# ^^^^^^^^^^^^^^^^^^^^
# We load the data into osl-dynamics using the Data class. See the :doc:`Loading Data tutorial <../tutorials_build/1-1_data_loading>` for further details.

from osl_dynamics.data import Data

data = Data("source_data")

#%%
# Trim data
# *********
# When we perform time-delay embedding we lose a few time points from the start and end of the data. Additionally, when we separate the data into sequences we lose time points from the end of the time series (that do not form a complete sequence). Consequently, we need to trim the source-space data to match the inferred state probabilities (`alpha`).

trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=200)  # needs to match values used to prepare the data and build the model

#%%
# If we didn't do any time-delay embedding and just need to remove time points lost due to separating into sequences we can use::
#
#     trimmed_data = data.trim_time_series(sequence_length=200)
#
# Load the mode mixing coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To calculate the mode spectra, we need the inferred mode mixing coefficients.

import pickle

alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

#%%
# `alpha` is a list of numpy arrays that contains a `(n_samples, n_modes)` time series, which is the mixing coefficients at each time point. Each item of the list corresponds to a subject.
#
# Let's double check the number of time points in inferred state probabilities match the source-space data.

for a, x in zip(alpha, trimmed_data):
    print(a.shape, x.shape)

#%%
# If the first dimension of these arrays don't match then the wrong value for `n_embeddings` or `sequence_length` was used when we trimmed the data.
#
# We use the mixing coefficients inferred by DyNeMo directly, not the re-weighted mixing coefficients (:func:`modes.reweight_alphas <osl_dynamics.inference.modes.reweight_alphas>`), which are for summary statistics.
#
# Calculating Subject/Mode Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using the mixing coefficients and source space data, we can calculate the spectral properties of each mode.
#
# Power spectra and coherences
# ****************************
# We want to calculate the power spectrum and coherence of each mode. We calculate a spectrogram (time-varying spectra) and fit it as a mixture of the mode spectra weighted by the mixing coefficients. We do this with the :func:`analysis.spectral.regression_spectra <osl_dynamics.analysis.spectral.regression_spectra>` function in osl-dynamics. Let's first run this function, then we'll discuss its output.

import numpy as np
from osl_dynamics.analysis import spectral

# Calculate regression spectra for each mode and subject (will take a few minutes)
f, psd, coh, w = spectral.regression_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
    return_weights=True,
)

#%%
# To understand the `f`, `psd` and `coh` numpy arrays it is useful to print their shape.

print(f.shape)
print(psd.shape)
print(coh.shape)
print(w.shape)

#%%
# We can see:
#
# - `f` is 1D and is the frequency axis.
# - `psd` is (subjects, 2, modes, channels, frequencies). The second dimension contains the regression coefficients (index 0) and intercept (index 1) for each subject. The intercept is the static (time-averaged) PSD. The coefficients are the change in the PSD from the static PSD for a one standard deviation increase in a mode's mixing coefficient: positive where a mode has more power than the static PSD and negative where it has less.
# - `coh` is (subjects, modes, channels, channels, frequencies). It is the change in the cross spectrum, normalized by the static power (change in coherence), for a one standard deviation increase in a mode's mixing coefficient. It includes changes in both coherence and power.
# - `w` contains the weight for each subject when calculating a group average.
#
# The coefficients of every subject are scaled by the same standard deviation, so the subject-specific spectra can be averaged and compared. See the :doc:`Spectral Analysis </user_guide/spectra>` page of the User Guide for how to interpret these outputs and what to plot.
#
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.

os.makedirs("results/spectra", exist_ok=True)
np.save("results/spectra/f.npy", f)
np.save("results/spectra/psd.npy", psd)
np.save("results/spectra/coh.npy", coh)
np.save("results/spectra/w.npy", w)
