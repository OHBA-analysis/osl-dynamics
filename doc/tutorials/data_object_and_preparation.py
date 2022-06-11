"""
The Data Object and Preparation
===============================
This is a tutorial that covers the use of the osl-dynamics Data object and the options for preparing a TensorFlow dataset to train a model.

"""

#%%
# We start by importing the necessary packages:

import numpy as np
from osl_dynamics import data, simulation

#%%
# Let's first simulate some data using a hidden Markov model with a multivariate normal observation model and save it to a file:

sim = simulation.HMM_MVN(
    n_samples=25600,
    n_states=5,
    n_channels=11,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
X = sim.time_series()
np.save("X.npy", X)

#%%
# Loading Data
# ^^^^^^^^^^^^
#
# Data can be loaded from a file in osl-dynamics with the Data class. The Data class can accept:
#
# - A string containing the path to a `.npy` or `.mat` file.
# - A list of strings containing paths to `.npy` or `.mat` files.
# - A path to a directory containing `.npy` files.
# - A numpy array can also be passed directly to Data.
#
# NOTE:
#
# - If a .mat file is passed, it must have a field called "X" that contains the data.
# - The data must be in the format (n_samples, n_channels).

training_data = data.Data("X.npy")

#%%
# To look at a summary of the data we can use:

print(training_data)

#%%
# The time series data can be accessed with the `time_series` method.

X = training_data.time_series()

#%%
# If multiple a list of file paths was passed to Data, then the `time_series` method will return a list of numpy arrays.
# The concatenated time series will be returned if `concatenate=True` is passed.

#%%
# Preparing the Data
# ^^^^^^^^^^^^^^^^^^
#
# The Data class can be used to apply standard transformations to the data to prepare it for training.
# The `prepare` method can be used for this. To standardize the data we can call:

training_data.prepare()

#%%
# To perform additional operations to standardization, we can pass arguments to `prepare`.
# E.g. to calculate the amplitude envelope then standardization, we use:

training_data.prepare(amplitude_envelope=True)

#%%
# Calling `training_data.time_series()` will now return the prepared data. The original data is still accessible with the `raw_data` attribute.

#%%
# Saving Prepared Data
# ^^^^^^^^^^^^^^^^^^^^
#
# It is sometimes useful to save prepared data. This can be done with

training_data.save("prepared_data")

#%%
# This will create a directory called `prepared_data` with `.npy` files containing the prepared data and a pickle file with preparation settings.
# The prepared data can be loaded in another script with:

training_data = Data("prepared_data")

#%%
# Creating TensorFlow Datasets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The Data object has a `dataset` method to created a batched TensorFlow dataset that can be used to train a model.
#

# Specify hyperparameters
sequence_length = 200
batch_size = 32

# Create a TensorFlow dataset for model training
training_dataset, validation_dataset = training_data.dataset(
    sequence_length,
    batch_size,
    shuffle=True,
    validation_split=0.1,  # if this is not passed, only the training_dataset is returned
)

# Create an unshuffled TensorFlow dataset for model evaluation
prediction_dataset = training_data.dataset(sequence_length, batch_size, shuffle=False)
