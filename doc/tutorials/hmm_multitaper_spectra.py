"""
HMM: Multitaper Spectra
=======================

In this tutorial we calculate subject and state-specific multitaper spectra using a trained HMM and **source space** data. Here, we should **not** use the prepared (e.g. TDE-PCA) data.

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/ktjzn>`_ for the expected output.
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

# Download the dataset (approximately 720 MB)
get_data("notts_mrc_meguk_glasser", rename="source_data")

#%%
# Load the source data
# ^^^^^^^^^^^^^^^^^^^^
# We load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.


from osl_dynamics.data import Data

data = Data("source_data", n_jobs=4)
print(data)

#%%
# Trim data
# *********
# If we trained on time-delay embedded data we lose a few time points from the start and end of the data. Additionally, when we separate the data into sequences we lose time points from the end of the time series (that do not form a complete sequence). Consequently, we need to trim the source-space data to match the inferred state probablities (`alpha`).


trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=1000)  # needs to match values used to prepare the data and build the model

#%%
# If we didn't do any time-delay embedding and just need to remove time points lost due to separating into sequences we can use::
#
#     trimmed_data = data.trim_time_series(sequence_length=1000)
#
# Load the state probabilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To calculate the multitaper spectra, we need the inferred state probabilities.


import pickle

alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

#%%
# `alpha` is a list of numpy arrays that contains a `(n_samples, n_states)` time series, which is the state probability at each time point. Each item of the list corresponds to a subject.
#
# Let's double check the number of time points in inferred state probabilities match the source-space data.


for a, x in zip(alpha, trimmed_data):
    print(a.shape, x.shape)

#%%
# If the first dimension of these arrays don't match then the wrong value for `n_embeddings` or `sequence_length` was used when we trimemd the data.
#
# Calculating Subjects/State Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Power spectra and coherences
# ****************************
# We want to calculate the power spectrum and coherence of each state. This is done by using standard calculation methods (in our case the multitaper for spectrum estimation) to the time points identified as belonging to a particular state. The `analysis.spectra.multitaper_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.multitaper_spectra>`_ function does this for us. Let's first run this function, then we'll discuss its output.


from osl_dynamics.analysis import spectral

# Calculate multitaper spectra for each state and subject (will take a few minutes)
f, psd, coh, w = spectral.multitaper_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    return_weights=True,
    n_jobs=4,
)

#%%
# To understand the `f`, `psd` and `coh` numpy arrays it is useful to print their shape.


print(f.shape)
print(psd.shape)
print(coh.shape)
print(w.shape)

#%%
# We can see the `f` array is 1D, it corresponds to the frequency axis for each spectra. The `psd` array corresponds to the PSDs and is (subjects, states, channels, frequencies) and the `coh` array corresponds to the pairwise coherences and is (subjects, states, channels, channels, frequencies). `w` is an array containing weights for each subject when calculating a group average.
#
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.


import numpy as np

os.makedirs("results/spectra", exist_ok=True)
np.save("results/spectra/f.npy", f)
np.save("results/spectra/psd.npy", psd)
np.save("results/spectra/coh.npy", coh)
np.save("results/spectra/w.npy", w)

