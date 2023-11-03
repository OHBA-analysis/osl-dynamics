"""
HMM: Training on Real Data
==========================

In this tutorial we will train a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:

1. Getting the Data
2. Fitting an HMM
3. Getting the Inferred Parameters
4. Calculating State Spectra

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/6kqa3>`_ for the expected output.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# We will use eyes open resting-state data that has already been source reconstructed. This dataset is:
#
# - From 10 subjects.
# - Parcellated to 42 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_. Note, `osfclient` must be installed. This can be done in jupyter notebook by running::
#
#    !pip install osfclient

import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 113 MB)
get_data("notts_rest_10_subj")

# List the contents of the downloaded directory
os.listdir("notts_rest_10_subj")

#%%
# Load the data
# *************
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

# Load data
training_data = Data("notts_rest_10_subj")

# Display some summary information
print(training_data)

#%%
# Fitting an HMM
# ^^^^^^^^^^^^^^
# Now we have loaded the data, let's see how to fit an HMM to it. In this tutorial, we want to study the spectral properties of dynamic networks. To do this we will use an approach known as time-delay embedding (TDE) to capture the spectral properties of each network in the model. This is commonly referred to as fitting a TDE-HMM.
#
# Prepare the data
# ****************
# It is straightforward to prepare TDE data (with PCA) using the Data class. See the `Data Preparation tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_preparation.html>`_ for further details. Let's prepare TDE-PCA data with 15 embeddings and 80 PCA components.

methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
training_data.prepare(methods)

#%%
# If we now print the Data object, we see the number of channels is the number of PCA components:

print(training_data)

#%%
# Notice, the total number of samples (`n_samples`) has also changed. This is due to the time-delay embedding. We lose `n_embeddings // 2` time points from either end of the time series for each subject. I.e. each subject has lost the first 7 and last 7 data points. We can see we have lost 726000 - 725860 = 140 = 10 * 14 data points. We can double check this by printing the shape of the prepared data for each subject and the shape of the original data.

# Get the original data time series
original_data = training_data.time_series(prepared=False)

# Get the prepared data time series
prepared_data = training_data.time_series()

for i in range(training_data.n_arrays):
    print(original_data[i].shape, prepared_data[i].shape)

#%%
# We can see each subject has lost 14 data points. You will always lose a total of `n_embeddings - 1` data points when performing time-delay embedding.
#
# The Config object
# *****************
# Now we have prepared the data, let's build a model to train. To do this we first need to specify the `Config object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Config>`_ for the HMM. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
#
# The important hyperparameters to specify are:
#
# - `n_states`, the number of states. Unfortunately, this is a hyperparameters that must be pre-specified. We advise starting with something between 6-14 and making sure any results based on the HMM are not critically sensitive to the choice for `n_states`. In this tutorial, we'll use 8 states.
# - `sequence_length` and `batch_size`. You want a large sequence length/batch size for fast training. However, you will find that holding large batches of long sequences requires a lot of memory. Therefore, you should pick the largest values that you can hold in memory. We find a sequence length of 1000-2000 and batch size of 8-64 works well for real neuroimaging data.
# - `learn_means` and `learn_covariances`. Typically, if we train on amplitude envelope data we set `learn_means` and `learn_covariances` to `True`, whereas if you're training on time-delay embedded/PCA data, then we only learn the covariances, i.e. we set `learn_means=False`.
# - `learning_rate`. On large datasets, we find a lower learning rate leads to a lower final loss. We recommend a value between 1e-2 and 1e-4. We advise training a few values and seeing which produces the lowest loss value.
# - `n_epochs`, the number of epochs. This is the number of times you loop through the data. We recommend a value between 15-40. You can look at the loss as a function of epochs (see below) to see when the model has stopped improving. You can use this as an indicator for when you can stop training.
#
# In general, you can use the final loss value (lower is better) to select a good set of hyperparameters.

from osl_dynamics.models.hmm import Config

# Create a config object
config = Config(
    n_states=8,
    n_channels=80,
    sequence_length=1000,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=1e-3,
    n_epochs=10,  # for the purposes of this tutorial we'll just train for a short period
)

#%%
# Building the model
# ******************
# With the Config object, we can build a model.

from osl_dynamics.models.hmm import Model

# Initiate a Model class and print a summary
model = Model(config)
model.summary()

#%%
# Training the model
# ******************
# Note, this step can be time consuming. Training this model on 10 subjects an M1 Macbook Air takes ~1 minute per epoch, which leads to a total training time of less than 15 minutes.
#
# **Initialization**
#
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the state means/covariances helps find a good explanation. The `hmm.Model <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Model>`_ class has a few helpful methods for initialization. When training on real data, we recommend using the `random_state_time_course_initialization`, let's do this. Usually 3 initializations is enough and you only need to train for a short period, we will use a single epoch.

init_history = model.random_state_time_course_initialization(training_data, n_epochs=1, n_init=3)

#%%
# The `init_history` variable is `dict` that contains the training history (`rho`, `lr`, `loss`) for the best initialization.
#
# **Full training**
#
# Now, we have found a good initialization, let's do the full training of the model. We do this using the `fit` method.

history = model.fit(training_data)

#%%
# The `history` variable containing the training history of the `fit` method.
#
# Saving a trained model
# **********************
# As we have just seen, training a model can be time consuming. Therefore, it is often useful to save a trained model so we can load it later. We can do this with the `save` method.

model.save("results/model")

#%%
# This will automatically create a directory containing the trained model weights and config settings used. Note, should we wish to load the trained model we can use::
#
#     from osl_dynamics.models import load
#
#     # Load the trained model
#     model = load("trained_model")
#
# Getting the Inferred Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We're interested in the hidden state time course inferred with the HMM. The HMM performs Bayesian inference of the state time course, i.e. it learns the probability at each time point of data belonging to a particular state. We can get the state probability time course using the `get_alpha` method of the HMM (equivalent to the Matlab HMM-MAR parameter `gamma`).

alpha = model.get_alpha(training_data)

#%%
# `alpha` is a list of numpy arrays that contains a `(n_samples, n_states)` time series, which is the probability of each state at each time point. Each item of the list corresponds to a subject. We can further understand the `alpha` list by printing the shape of its items.

for a in alpha:
    print(a.shape)

#%%
# Notice, the number of samples in the state probability time series for each subject does not match the number of samples in the prepared data for each subject. Printing them side by side, we have:

for a, x in zip(alpha, prepared_data):
    print(a.shape, x.shape)

#%%
# This is due to the training process which first segments the data for each subject into sequences of length `sequence_length` and then groups them into groups of size `batch_size`. If there are extra data points in a subject's time series that do not fit into a sequence, they are dropped. Looking at the first subject, we see the state probability time series (`alpha[0]`) has 74000 samples, but the prepared data has 74486 samples, this means the first subject was segmented into `74486 // sequence_length = 74486 // 1000 = 74` sequences, which will give `74 * sequence_length = 74 * 1000 = 74000` samples in the state probability time course.
#
# We often want to align the inferred state probability to the original (unprepared) data. This can be done using the `get_training_time_series` method of a model. Note, this method is defined in the `model base class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/mod_base/index.html>`_, which is inherited by the `hmm.Model` class. Let's use this function to get the original source reconstructed data that each time point in the state probability time course corresponds to.

data = model.get_training_time_series(training_data, prepared=False)

#%%
# `data` is a list of numpy arrays, one for each subject. Now if print the shape of the state probability time course and training data, we'll see they match.

for a, x in zip(alpha, data):
    print(a.shape, x.shape)

#%%
# We don't want to have to keep loading the model to get the inferred state probabilities, instead let's just save the inferred state probabilties.

import pickle

# Save the python lists as a pickle files
os.makedirs("results/data", exist_ok=True)
pickle.dump(alpha, open("results/data/alpha.pkl", "wb"))

#%%
# Calculating State Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Using the inferred state probabilities and training data (unprepared) we can calculate the spectral properties of each state.
#
# Power spectra and coherences
# ****************************
# We want to calculate the power spectrum and coherence of each state. This is done by using standard calculation methods (in our case the multitaper for spectrum estimation) to the time points identified as belonging to a particular state. The `analysis.spectra.multitaper_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.multitaper_spectra>`_ function does this for us. Let's first run this function, then we'll discuss its output. The arguments we need to pass to this function are:
#
# - `data`. This is the source reconstructed data aligned to the state time course.
# - `alpha`. This is the state time course or probabilities (either can be used). Here we'll use the state probabilities.
# - `sampling_frequency` in Hz.
# - `time_half_bandwidth`. This is a parameter for the multitaper, we suggest using `4`.
# - `n_tapers`. This is another parameter for the multitaper, we suggest using `7`.
# - `frequency_range`. This is the frequency range we're interested in.

from osl_dynamics.analysis import spectral

# Calculate multitaper spectra for each state and subject (will take a few minutes)
f, psd, coh = spectral.multitaper_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
)

#%%
# Note, there is a `n_jobs` argument that can be used to calculate the multitaper spectrum for each subject in parallel.
#
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.

import numpy as np

np.save("results/data/f.npy", f)
np.save("results/data/psd.npy", psd)
np.save("results/data/coh.npy", coh)

#%%
# To understand the `f`, `psd` and `coh` numpy arrays it is useful to print their shape.

print(f.shape)
print(psd.shape)
print(coh.shape)

#%%
# We can see the `f` array is 1D, it corresponds to the frequency axis. The `psd` array is (subjects, states, channels, frequencies) and the `coh` array is (subjects, states, channels, channels, frequencies).
#
# Wrap Up
# ^^^^^^^
# - We have shown how to prepare time-delay embedded/PCA data for training an HMM.
# - We have trained and saved an HMM and its inferred parameters (state probabilities).
# - We have calculated the spectral properties of each state.
