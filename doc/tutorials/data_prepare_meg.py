"""
Preparing M/EEG Data
====================

In this tutorial we will go through common ways to prepare M/EEG data. This tutorial covers:

1. Downloading example data
2. Time-Delay Embedding (and Principal Component Analysis)
3. Amplitude Envelope
4. Saving and Loading Prepared Data

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/m83ze>`_ for the expected output.
"""

#%%
# Download the dataset
# ^^^^^^^^^^^^^^^^^^^^
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.


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
# Time-delay embedding (TDE) is a process of augmenting a time series with extra channels. These extra channels are time-lagged versions of the original channels. We do this to add extra entries to the covariance matrix of the data which are sensitive to the frequency of oscillations in the data.
#
# Here, we demonstrate how to apply this process to data. For further details, including the impact of different parameters, see the `TDE tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_time_delay_embedding.html>`_. 
#
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
# When we perform PCA a `pca_components` attribute is added to the Data object. This is the `(n_raw_channels, n_pca_components)` shape array that is used to perform PCA.


pca_components = data.pca_components
print(pca_components.shape)

#%%
# Amplitude envelope
# ******************
# Another approach for preparing data is to calculate the amplitude envelope. Here, we obtain a time series that characterises the amplitude of oscilations.
#
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
