"""
DyNeMo: Regression Spectra
==========================

In this tutorial we calculate subject and mode-specific spectra using a trained DyNeMo model and **source space** data. Here, we should **not** use the prepared (e.g. TDE-PCA) data.

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/zwyaj>`_ for the expected output.
"""

#%%
# Download source data
# ^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data from `OSF <https://osf.io/by2tc/>`_.


import os

def get_data(name, rename):
    if rename is None:
        rename = name
    if os.path.exists(rename):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 720 GB)
get_data("notts_mrc_meguk_glasser", rename="source_data")

#%%
# Load the source data
# ^^^^^^^^^^^^^^^^^^^^
# We load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.


from osl_dynamics.data import Data

data = Data("source_data", n_jobs=4)

#%%
# Trim data
# *********
# When we perform time-delay embedding we lose a few time points from the start and end of the data. Additionally, when we separate the data into sequences we lose time points from the end of the time series (that do not form a complete sequence). Consequently, we need to trim the source-space data to match the inferred state probablities (`alpha`).


trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=100)  # needs to match values used to prepare the data and build the model

#%%
# If we didn't do any time-delay embedding and just need to remove time points lost due to separating into sequences we can use::
#
#     trimmed_data = data.trim_time_series(sequence_length=100)
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
# If the first dimension of these arrays don't match then the wrong value for `n_embeddings` or `sequence_length` was used when we trimemd the data.
#
# Reweight the mixing coefficients
# ********************************
# The mixing coefficients inferred by DyNeMo do not account for the 'size' of each mode covariance. Before calculate the mode spectra, we reweight the mixing coefficients using the trace of each mode covariance. First, let's load the inferred mode covariances.


import numpy as np

covs = np.load("results/inf_params/covs.npy")

#%%
# Now, let's reweight the mixing coefficients.


from osl_dynamics.inference import modes

alpha = modes.reweight_alphas(alpha, covs)

#%%
# Calculating Subject/Mode Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using the reweights mixing coefficients and source space data, we can calculate the spectral properties of each mode.
#
# Power spectra and coherences
# ****************************
# We want to calculate the power spectrum and coherence of each mode. A linear regression approach where we regress the spectrogram (time-varying spectra) with the inferred mixing coefficients. We do this with the `analysis.spectral.regression_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.regression_spectra>`_ function in osl-dynamics. Let's first run this function, then we'll discuss its output.


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
    n_jobs=2,
)

#%%
# To understand the `f`, `psd` and `coh` numpy arrays it is useful to print their shape.


print(f.shape)
print(psd.shape)
print(coh.shape)
print(w.shape)

#%%
# We can see the `f` array is 1D, it corresponds to the frequency axis. The `psd` array corresponds to the PSDs and is (subjects, 2, states, channels, frequencies), the second dimension corresponds to the regression coefficients and intercept for each subject. The `coh` array corresponds to the pairwise coherence and is (subjects, states, channels, channels, frequencies). `w` is an array containing weights for each subject when calculating a group average.
#
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.


os.makedirs("results/spectra", exist_ok=True)
np.save("results/spectra/f.npy", f)
np.save("results/spectra/psd.npy", psd)
np.save("results/spectra/coh.npy", coh)
np.save("results/spectra/w.npy", w)

