"""
Preparing Data
==============
 
In this tutorial we demonstrate how to prepare data for training an osl-dynamics model. This tutorial covers:
 
1. Getting Example Data and Loading the Data
2. Preparing Time-Delay Embedded, Principal Component Analysis (TDE-PCA) Data
3. Preparing Amplitude Envelope Data
4. Saving and Loading Prepared Data

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/ync38>`_ for the expected output.
"""

#%%
# Getting Example Data and Loading the Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Download the dataset
# ********************
# 
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

# Download the dataset (approximate 52 MB)
get_data("example_loading_data")

# List the contents of the downloaded directory containing the dataset
print("Contents of example_loading_data:")
os.listdir("example_loading_data")

#%%
# Loading the data
# ****************
# 
# Now, let's load the example data into osl-dynamics. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("example_loading_data/numpy_format")
print(data)

#%%
# We can see we have data for two subjects.
# 
# Preparing Time-Delay Embedded, Principal Component Analysis (TDE-PCA) Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Time-delay embedding
# ********************
# 
# Next, we will demonstrate how we prepare the data for training a model. A common approach used to model the spectral properties of time series data is to apply time-delay embedding (TDE). This is a process of adding extra channels to the time series containing time lagged versions of the original data. By doing this we encode the autocorrelation function into the covariance matrix of the data. The autocorrelation function characterises the spectral properties (i.e. frequency specific characteristics) of a time series.
# 
# Principal component analysis
# ****************************
# 
# After time embedding we are often left with a very high-dimensional time series. Therefore, to make this time series more managable we apply principal component analysis (PCA) to reduce the dimensionality.
# 
# The prepare method
# ******************
# 
# The `Data class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ has a method for preparing the data called `prepare`. This is an 'in-place' operator, it does not return anything.
# 
# To time-delay embed the data, we just need to pass the `n_embeddings` argument. Additionally, we can pass the `n_pca_components` argument to the `prepare` method to perform PCA. Note, if you just want to perform TDE then you can omit the `n_pca_component` argument, similarly, the `n_embeddings` argument can be omitted if you'd just like to perform PCA. For MEG data, we typically use `n_embeddings=15`, which corresponds to adding channels with -7, -6, ..., 0, 6, 7 lags, and `n_pca_components=80`.
# 
# **Note, standardization (removing the mean and normalizing each channel to unit variance) is always performed as the last step in the `prepare` method.**
# 
# To understand TDE-PCA a bit better let's start by just performing TDE.

data.prepare(n_embeddings=15)
print(data)

#%%
# We can see the `n_samples` attribute of our Data object has changed from 147500 to 147472. We have lost 28 samples. This is due to the TDE. We lose `n_embeddings // 2` data points from each end of the time series for each subject. In other words, with `n_embeddings=15`, we lose the first 7 and last 7 data points from each subject. We lose a total of 28 data points because we have 2 subjects.
# 
# We also see the number of channels has increased from 42 to 630, this is because we've added an additional `n_embeddings - 1 = 14` channels for every original channel we had.
# 
# **Note, if we call the `prepare` method again, it will start from the original raw data.**
# 
# Now, let's prepare TDE-PCA data.

data.prepare(n_embeddings=15, n_pca_components=80)
print(data)

#%%
# We can now see the number of channels is the number of PCA components as expected.
# 
# When we perform PCA a `pca_components` attribute is added to the Data object. This is the `(n_raw_channels, n_pca_components)` shape array that is used to perform PCA.

pca_components = data.pca_components
print(pca_components.shape)

#%%
# Accessing the prepared and raw data
# ***********************************
# 
# After preparing the data, the `time_series` method will return the prepared data.

ts = data.time_series()  # ts is a list of subject-specific numpy arrays
print(ts[0].shape)

#%%
# We can see the shape of the data indicates it is the prepared data.
# 
# The raw data is still accessible via the `raw_data` attribute.

raw = data.raw_data  # raw is a list of subject-specific numpy arrays
print(raw[0].shape)

#%%
# Preparing Amplitude Envelope Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Another common approach is to train a model on amplitude envelope data. The `prepare` method can also be used for this. Amplitude envelope data can be calculated using the `amplitude_envelope` argument.

data.prepare(amplitude_envelope=True)
print(data)

#%%
# We can see when we prepare amplitude envelope data, the number of samples and channels does not change.
# 
# It is also common to apply a sliding window filter to smooth the data after calculating the amplitude envelope. We can do this using the `n_window` argument, which is the number of samples for the sliding window. The sliding window uses a boxcar windowing function. For MEG data, we typically use `n_window=6`. Let's prepare amplitude envelope data followed by a sliding window.

data.prepare(amplitude_envelope=True, n_window=6)
print(data)

#%%
# Now we see we have lost 10 data points. In this case we have lost 5 data points from the end of the time series for each subject. We can verify this by printing the shape of each subject's time series.

# Get the time series for each subject
ts = data.time_series()
for x in ts:
    print(x.shape)

#%%
# Saving and Loading Prepared Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Saving prepared data
# ********************
# 
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
# 
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
# 
# - We have shown how to prepare TDE-PCA and amplitude envelope data.
# - We have shown how to save the prepared data and how to load it.
