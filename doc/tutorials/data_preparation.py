"""
Preparing Data
==============

In this tutorial we demonstrate how to prepare data for training an osl-dynamics model. This tutorial covers:

1. Time-Delay Embeddeding
2. Amplitude Envelope
3. Preparing Real Data: The Prepare Method
4. Saving and Loading Prepared Data

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/ync38>`_ for the expected output.
"""

#%%
# Time-Delay Embedding
# ^^^^^^^^^^^^^^^^^^^^
# Time-delay embedding (TDE) is a process of augmenting a time series with extra channels. These extra channels are time-lagged versions of the original channels. We do this to add extra entries to the covariance matrix of the data which are sensitive to the frequency of oscillations in the data. To understand this better, let's simulate some sinusoidal data.

import numpy as np
import matplotlib.pyplot as plt

# Simulate data
n = 10000
t = np.arange(n) / 200  # we're using a sampling frequency of 200 Hz
x = np.array([
    np.sin(2 * np.pi * 10 * t),  # 10 Hz sine wave
    1.5 * np.sin(2 * np.pi * 20 * t),  # 20 Hz sine wave
])

# Plot first 0.1 s
plt.plot(t[:20], x[0,:20], label="Channel 1")
plt.plot(t[:20], x[1,:20], label="Channel 2")
plt.xlabel("Time (s)")
plt.legend()

#%%
# We can see a 10 Hz and 20 Hz sine wave. Let's plot the covariance of this data.

cov = np.cov(x)

plt.matshow(cov)
plt.colorbar()

#%%
# The covariance here is a 2x2 matrix because we have 2 channels. The diagonal of this matrix is the variance and reflects the amplitude of each sine wave. The off-diagonal elements reflect the covariance between channels. In this example the covariance between the channels is close to zero. Let's see what happens to the covariance matrix when we TDE the data.

from osl_dynamics.data import Data

# First load the data into osl-dynamics
data = Data(x.T)
print(data)

# Perform time-delay embedding
data.tde(n_embeddings=5)

#%%
# See the `Data loading tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details regarding how to load data using the `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ class. In the above code we chose `n_embeddings=5`. This means for every original channel, we add `n_embeddings - 1 = 4` extra channels. In our two channel example, the operation we do is:
#
# [x(t), y(t)] -> [x(t-2), x(t-1), x(t), x(t+1), x(t+2), y(t-2), y(t-1), y(t), y(t+1), y(t+2)].
#
# We should expect a total of `n_embeddings * 2` channels, in our example this is `5 * 2 = 10`. We can verify this by printing the Data object.

print(data)

#%%
# We can see we have 10 channels as expected. Note, we have also lost `n_embeddings - 1 = 4` time points (we have 9996 samples when originally we simulated 10000). This is because we don't have the full window to TDE the time points at the start and end of the time series.
#
# Let's look at the covariance of the TDE data.

cov_tde = np.cov(data.time_series(), rowvar=False)

plt.matshow(cov_tde)
plt.colorbar()

#%%
# This covariance matrix is 10x10 because we have 10 channels. Some elements in this matrix are the covariance of a channel with a time-lagged version of itself - this quantity is known as the auto-correlation function. We can extract an estimate of the auto-correlation function (ACF) by taking vaules from this covariance matrix. osl-dynamics has a function we can use for this: `analysis.modes.autocorr_from_tde_cov <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.autocorr_from_tde_cov>`_.

from osl_dynamics.analysis import modes, spectral

# Extract auto (and cross) correlation functions from the covariance
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=5)
print(acf.shape)  # channels x channels x time lags

#%%
# The ACF and power spectral density (PSD) form a Fourier pair. This means we can calculate an estimate of the PSD of each channel by Fourier transforming the ACF. Let's do this using the `analysis.spectral.autocorr_to_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.autocorr_to_spectra>`_ function in osl-dynamics.

# Calculate power spectral density by Fourier transforming the auto-correlation function
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)
print(psd.shape)  # channels x channels x frequency

# Plot
plt.plot(f, psd[0,0], label="Channel 1")
plt.plot(f, psd[1,1], label="Channel 2")
plt.legend()

#%%
# We can see the 20 Hz peak in the channel 2, which corresponds well to what we simulated. However,  we we're able to resolve the 10 Hz peak. This was because we didn't use enough lags to resolve the 10 Hz peak. Note, we can see some ringing, this is due to padding the ACF with zeros (to obtain an interger multiple of 2) before calculating the Fourier transform, we can change the padding via the `nfft` argument to `analysis.spectral.autocorr_to_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.autocorr_to_spectra>`_.
#
# Let's try again with more lags - this will mean we evaluate the ACF for a greater window of time lags, this will results in a higher resolution PSD.

# Redo the TDE on the original data
data.tde(n_embeddings=11, use_raw=True)
print(data)

# Calculate TDE covariance, ACF and PSD
cov_tde = np.cov(data.time_series(), rowvar=False)
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=11)
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].matshow(cov_tde)
ax[1].plot(f, psd[0,0], label="Channel 1")
ax[1].plot(f, psd[1,1], label="Channel 2")
ax[1].legend()

#%%
# We can see we're now able to better model the 10 Hz sine wave in channel 1. We can also see what happens if we change the frequency of the sine wave for channel 1. Let's see what happens if we simulate a 30 Hz sine wave for channel 1.

# Simulate new data
x = np.array([
    np.sin(2 * np.pi * 30 * t),  # 30 Hz sine wave
    1.5 * np.sin(2 * np.pi * 20 * t),  # 20 Hz sine wave
])
data = Data(x.T)

# TDE
data.tde(n_embeddings=11)

# Calculate TDE covariance, ACF and PSD
cov_tde = np.cov(data.time_series(), rowvar=False)
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=11)
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].matshow(cov_tde)
ax[1].plot(f, psd[0,0], label="Channel 1")
ax[1].plot(f, psd[1,1], label="Channel 2")
ax[1].legend()

#%%
# We can see the covariance of the TDE data has changed to reflect the frequency of data. The above example shows how TDE leads to covariance matrices that are sensitive to oscillatory frequencies in the original data and how the number of embeddings relates to the frequency resolution that can be modelled.
#
# Amplitude Envelope
# ^^^^^^^^^^^^^^^^^^
# Another approach for preparing data is to calculate the amplitude envelope. Here, we obtain a time series that characterises the amplitude of oscilations. To understand this operation, let's calculate the amplitude envelope of a modulated sine wave.

# Simulate data
x = np.cos(2 * np.pi* 3 * t) * np.sin(2 * np.pi * 20 * t)

# Load into osl-dynamics
data = Data(x.T)

# Calculate amplitude envelope
data.amplitude_envelope()
ae = data.time_series()

# Plot
plt.plot(t[:40], x[:40], label="Original")
plt.plot(t[:40], ae[:40], label="Amp. Env.")
plt.xlabel("Time (s)")
plt.legend()

#%%
# We can see we lose the frequency and phase information and only retain the amplitude of the oscillation.
#
# Preparing Real Data: The Prepare Method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section we'll give examples of how to prepare data using the `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ class in osl-dynamics. We will use real data for this.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_. Note, `osfclient` must be installed. This can be done in jupyter notebook by running::
#
#     !pip install osfclient

import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximate 88 MB)
get_data("example_loading_data")

# List the contents of the downloaded directory containing the dataset
print("Contents of example_loading_data:")
os.listdir("example_loading_data")

#%%
# Loading the data
# ****************
# Now, let's load the example data into osl-dynamics. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("example_loading_data/numpy_format")
print(data)

#%%
# We can see we have data for two subjects.
#
# The prepare method
# ^^^^^^^^^^^^^^^^^^
# The `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ class has many methods available for manipulating the data. See the docs for a full list. We will describe two common approaches for preparing the data:
#
# - TDE-PCA
# - Amplitude Envelope.
#
# We will use the `prepare` method to do this. This method takes a `dict` containing method names to call and arguments to pass to them.
#
# TDE-PCA
# *******
# Performing TDE often results in a very large number of channels. Consequently, Principal Component Analysis (PCA) is often used to reduce the number of channels. Both TDE and PCA can be done in one step using the `tde_pca` method. We often also want to standardize (z-transform) the data before training a model. Both of these steps can be done with the `prepare` method.

data = Data("example_loading_data/numpy_format")
print(data)

methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
data.prepare(methods)
print(data)

#%%
# We can see the `n_samples` attribute of our Data object has changed from 147500 to 147472. We have lost 28 samples. This is due to the TDE. We lose `n_embeddings // 2` data points from each end of the time series for each subject. In other words, with `n_embeddings=15`, we lose the first 7 and last 7 data points from each subject. We lose a total of 28 data points because we have 2 subjects.
#
# We also see the number of channels has increased from 42 to 630, this is because we've added an additional `n_embeddings - 1 = 14` channels for every original channel we had.
#
# 15 embeddings and 80 PCA components have been found to work well on multiple datasets and would be a good default to use.
#
# When we perform PCA a `pca_components` attribute is added to the Data object. This is the `(n_raw_channels, n_pca_components)` shape array that is used to perform PCA.

pca_components = data.pca_components
print(pca_components.shape)

#%%
# Amplitde envelope
# *****************
# To prepare amplitude envelope data, we perform a few steps. It's common to first filter a frequency range of interest before calculating the amplitude envelope. Calculating a moving average has also been found to help smooth the data. Finally, standardization is always recommended for models in osl-dynamics.

data = Data("example_loading_data/numpy_format", sampling_frequency=200)
print(data)

methods = {
    "filter": {"low_freq": 7, "high_freq": 13},  # study the alpha-band
    "amplitude_envelope": {},
    "moving_average": {"n_window": 5},
    "standardize": {},
}
data.prepare(methods)
print(data)

#%%
# When we apply a moving average we lose a few time points from each end of the time series for each subject. We lose `n_window // 2` time points, for `n_window=5` we lose 2 time points from the start and end. This totals to 4 time points lost for each subject. We can see this looking at the `n_samples` attribute of the Data object, where we have lost 4 time points for each subject, totalling to 8 time points. 
#
# Accessing the prepared and raw data
# ***********************************
# After preparing the data, the `Data.time_series` method will return the prepared data.

ts = data.time_series()  # ts is a list of subject-specific numpy arrays
print(ts[0].shape)

#%%
# We can see the shape of the data indicates it is the prepared data.
#
# The raw data is still accessible by passing `prepared=False` to the `Data.time_series` method.

raw = data.time_series(prepared=False)  # raw is a list of subject-specific numpy arrays
print(raw[0].shape)

#%%
# Saving and Loading Prepared Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Saving prepared data
# ********************
# For large datasets, preparing the data can sometimes be time consuming. In this case it is useful to save the data after preparing it. Then we can just load the prepared data before training a model. To save prepared data we can use the `save` method. We simply need to pass the output directory to write the data to.

data.save("prepared_data")

#%%
# This method has created a directory called `prepared_data`. Let's list its contents.

os.listdir("prepared_data")

#%%
# We can see each subject's data is saved as a numpy file and there is an additional pickle (`.pkl`) file which contains information regarding how the data was prepared.
#
# Loading prepared data
# *********************
# We can load the prepared data by simply passing the path to the directory to the Data class.

data = Data("prepared_data")
print(data)

#%%
# We can see from the number of samples it is the amplitude envelope data that we previously prepared.
#
# Note, if we saved data that included PCA in preparation. The `pca_components` attribute will be loaded from the pickle file when we load data using the Data class.
#
# Wrap Up
# ^^^^^^^
# - We have discussed how TDE and calculating an amplitude envelope affects the data.
# - We have shown how to prepare TDE-PCA and amplitude envelope data.
# - We have shown how to save the prepared data and how to load it.
