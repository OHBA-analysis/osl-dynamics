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

# Simulate data:
# - Channel 1: 10 Hz sine wave
# - Channel 2: 20 Hz sine wave
# - Channel 3: 20 Hz sine wave synchronised with channel 2 but with different amplitude
n = 10000
t = np.arange(n) / 200  # we're using a sampling frequency of 200 Hz
p = np.random.uniform(0, 2 * np.pi, size=(2,))  # random phases
x = np.array([
    1.0 * np.sin(2 * np.pi * 10 * t + p[0]),
    1.5 * np.sin(2 * np.pi * 20 * t + p[1]),
    0.6 * np.sin(2 * np.pi * 20 * t + p[1]),  # same phase as channel 2
])

# Add some noise
x += np.random.normal(0, 0.1, size=x.shape)

# Plot first 0.2 s
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(t[:40], x[0,:40], label="Channel 1")
ax[1].plot(t[:40], x[1,:40], label="Channel 2")
ax[2].plot(t[:40], x[2,:40], label="Channel 3")
ax[0].set_ylabel("Channel 1")
ax[1].set_ylabel("Channel 2")
ax[2].set_ylabel("Channel 3")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()

#%%
# Let's plot the covariance of this data.

cov = np.cov(x)

plt.matshow(cov)
plt.colorbar()

#%%
# The covariance here is a 3x3 matrix because we have 3 channels. The diagonal of this matrix is the variance and reflects the amplitude of each sine wave. The off-diagonal elements reflect the covariance between channels. In this example the covariance between the channels 1 and 2 is close to zero, however there is some covariance between channels 2 and 3 due to the phase synchronisation. Let's see what happens to the covariance matrix when we TDE the data.

from osl_dynamics.data import Data

# First load the data into osl-dynamics
data = Data(x, time_axis_first=False)
print(data)

# Perform time-delay embedding
data.tde(n_embeddings=5)

#%%
# See the `Data loading tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details regarding how to load data using the `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ class. In the above code we chose `n_embeddings=5`. This means for every original channel, we add `n_embeddings - 1 = 4` extra channels. In our three channel example, the operation we do is:
#
# .. math::
#     \begin{pmatrix} x(t) \\ y(x) \\ z(t) \end{pmatrix} \rightarrow \begin{pmatrix} x(t-2) \\ x(t-1) \\ x(t) \\ x(t+1) \\ x(t+2) \\ y(t-2) \\ y(t-1) \\ y(t) \\ y(t+1) \\ y(t+2) \\ z(t-2) \\ z(t-1) \\ z(t) \\ z(t+1) \\ z(t+2) \end{pmatrix}
#
# We should expect a total of `n_embeddings * 3` channels, in our example this is `5 * 3 = 15`. We can verify this by printing the Data object.

print(data)

#%%
# We can see we have 15 channels as expected. Note, we have also lost `n_embeddings - 1 = 4` time points (we have 9996 samples when originally we simulated 10000). This is because we don't have the full window to TDE the time points at the start and end of the time series.
#
# Let's look at the covariance of the TDE data.

cov_tde = np.cov(data.time_series(), rowvar=False)

plt.matshow(cov_tde)
plt.colorbar()

#%%
# This covariance matrix is 15x15 because we have 15 channels. The blocks on the diagonal of the above matrix represent the covariance of a channel with a time-lagged version of itself - this quantity is known as the **auto**-correlation function. Blocks on the off-diagonal represent the covariance of a channel with a time-lagged version of **another** channel - this quantity is known as the **cross**-correlation function.
#
# We can extract an estimate of the auto/cross-correlation function (A/CCF) by taking values from this covariance matrix. osl-dynamics has a function we can use for this: `analysis.modes.autocorr_from_tde_cov <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.autocorr_from_tde_cov>`_. This function will extract both ACFs and CCFs from a TDE covariance matrix.

from osl_dynamics.analysis import modes, spectral

# Extract A/CCFs from the covariance
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=5)
print(acf.shape)  # channels x channels x time lags

# Plot ACFs
plt.plot(tau, acf[0,0], label="Channel 1")
plt.plot(tau, acf[1,1], label="Channel 2")
plt.plot(tau, acf[2,2], label="Channel 3")
plt.xlabel("Time Lag (Samples)")
plt.ylabel("ACF (a.u)")
plt.legend()
plt.tight_layout()

#%%
# The diagonal of `acf` represents the ACF, e.g. `acf[0,0]` is the ACF for for channel 1 (indexed by 0). The off-diagonal of `acf` represent the CCF, e.g. `acf[0,1]` is the CCF for channel 1 and 2.
#
# The ACF and power spectral density (PSD) form a Fourier pair. This means we can calculate an estimate of the PSD of each channel by Fourier transforming the ACF. Let's do this using the `analysis.spectral.autocorr_to_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.autocorr_to_spectra>`_ function in osl-dynamics. Note, this function will also calculate cross PSD using the CCFs.

# Calculate PSD by Fourier transforming the ACF
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)
print(psd.shape)  # channels x channels x frequency

# Plot
plt.plot(f, psd[0,0], label="Channel 1")
plt.plot(f, psd[1,1], label="Channel 2")
plt.plot(f, psd[2,2], label="Channel 3")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (a.u)")
plt.legend()

#%%
# We can see the 20 Hz peak in the channel 2, which corresponds well to what we simulated. We also see a smaller 20 Hz peak for channel 3. However, we aren't able to resolve the 10 Hz peak we simulated for channel 1. This was because we didn't use enough lags to resolve the 10 Hz peak.
#
# Note, we can see some ringing, this is due to padding the ACF with zeros (to obtain an interger multiple of 2) before calculating the Fourier transform, we can change the padding via the `nfft` argument to `analysis.spectral.autocorr_to_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.autocorr_to_spectra>`_.
#
# Let's try again with more lags - this will mean we evaluate the ACF for a greater window of time lags, this will result in a higher resolution PSD.

# Redo the TDE on the original data
data.tde(n_embeddings=11, use_raw=True)
print(data)

# Calculate TDE covariance, ACF and PSD
cov_tde = np.cov(data.time_series(), rowvar=False)
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=11)
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
ax[0].matshow(cov_tde)

ax[1].plot(tau, acf[0,0])
ax[1].plot(tau, acf[1,1])
ax[1].plot(tau, acf[2,2])
ax[1].set_xlabel("Time Lag (Samples)")
ax[1].set_ylabel("ACF (a.u.)")

ax[2].plot(f, psd[0,0], label="Channel 1")
ax[2].plot(f, psd[1,1], label="Channel 2")
ax[2].plot(f, psd[2,2], label="Channel 2")
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("PSD (a.u.)")
ax[2].legend()
plt.tight_layout()

#%%
# We can see the ACF extends over a wider range and we're now able to better model the 10 Hz sine wave in channel 1. We can also see what happens if we change the frequency of the sine wave for channel 1. Let's simulate a 30 Hz sine wave for channel 1.

# Simulate new data
x = np.array([
    1.0 * np.sin(2 * np.pi * 30 * t + p[0]),  # 30 Hz
    1.5 * np.sin(2 * np.pi * 20 * t + p[1]),
    0.6 * np.sin(2 * np.pi * 20 * t + p[1]),  # same phase as channel 2
])
x += np.random.normal(0, 0.15, size=x.shape)
data = Data(x, time_axis_first=False)

# TDE
data.tde(n_embeddings=11)

# Calculate TDE covariance, ACF and PSD
cov_tde = np.cov(data.time_series(), rowvar=False)
tau, acf = modes.autocorr_from_tde_cov(cov_tde, n_embeddings=11)
f, psd, _ = spectral.autocorr_to_spectra(acf, sampling_frequency=200)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
ax[0].matshow(cov_tde)

ax[1].plot(tau, acf[0,0])
ax[1].plot(tau, acf[1,1])
ax[1].plot(tau, acf[2,2])
ax[1].set_xlabel("Time Lag (Samples)")
ax[1].set_ylabel("ACF (a.u.)")

ax[2].plot(f, psd[0,0], label="Channel 1")
ax[2].plot(f, psd[1,1], label="Channel 2")
ax[2].plot(f, psd[2,2], label="Channel 2")
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("PSD (a.u.)")
ax[2].legend()
plt.tight_layout()

#%%
# We can see the covariance of the TDE data, as well as the ACF/PSD has, changed to reflect the frequency of data. The above example shows how TDE leads to covariance matrices that are sensitive to oscillatory frequencies in the original data and how the number of embeddings relates to the frequency resolution that can be modelled.
#
# In addition to modelling the frequency of oscillations in an individual channel, we can also model phase synchronisation across channels using TDE data. We can see from the above TDE covariance matrix the off-diagonal block for channels 2 and 3 (row 10-20, column 20-30) shows some structure. Let's plot the CCF for each pair of channels.

plt.plot(tau, acf[0,1], label="Channel 1+2")
plt.plot(tau, acf[1,2], label="Channel 2+3")
plt.xlabel("Time Lag (Samples)")
plt.ylabel("CCF (a.u.)")
plt.legend()

#%%
# The structure in the CCF for channels 2 and 3 shows the time-lagged versions of these channels are correlated. Such structure arises from phase synchronisation. In contrast, we don't see any structure for the CCF between channels 1 and 2. This is because we simulated random phases for these channels.
#
# Let's plot the cross PSD for these channels.

plt.plot(f, psd[0,1], label="Channel 1+2")
plt.plot(f, psd[1,2], label="Channel 2+3")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Cross PSD (a.u)")
plt.legend()

#%%
# We can see the cross PSD indicates the frequencies which show phase synchronisation.
#
# In the above examples we have shown how the covariance of TDE data can capture the oscillatory properties (power spectra and frequency-specific coupling) of a time series.
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
# 15 embeddings and 80 PCA components have been found to work well on multiple datasets and would be a good default to use.
#
# When we perform PCA a `pca_components` attribute is added to the Data object. This is the `(n_raw_channels, n_pca_components)` shape array that is used to perform PCA.

pca_components = data.pca_components
print(pca_components.shape)

#%%
# Amplitude envelope
# ******************
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

os.listdir('prepared_data')

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
