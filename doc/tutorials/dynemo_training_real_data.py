"""
DyNeMo: Training on Real Data
=============================
 
In this tutorial we will train a Dynamic Network Modes (DyNeMo) model on resting-state source reconstructed MEG data. This tutorial covers:
 
1. Getting the Data
2. Fitting DyNeMo
3. Getting the Inferred Parameters
4. Calculating Mode Spectra

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/zb5pk>`_ for the expected output.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# 
# We will use eyes open resting-state data that has already been source reconstructed. This dataset is:
#
# - From 10 subjects.
# - Parcellated to 42 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# 
# Download the dataset
# ********************
# 
# We will download example data hosted on `OSF <https://osf.io/zxb6c/>`_. Note, `osfclient` must be installed. This can be done in jupyter notebook by running::
#
#     !pip install osfclient
#

import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch Dynamics/data/datasets/{name}.zip")
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
# 
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osf.io/ejxut>`_ for further details.

from osl_dynamics.data import Data

# Load data
training_data = Data("notts_rest_10_subj")

# Display some summary information
print(training_data)

#%%
# Fitting DyNeMo
# ^^^^^^^^^^^^^^
# 
# Now we have loaded the data, let's see how to fit a DyNeMo model to it. In this tutorial, we want to study the spectral properties of dynamic networks. To do this we will use an approach known as time-delay embedding (TDE) to capture the spectral properties of each network in the model.
# 
# Prepare the data
# ****************
# 
# It is straightforward to prepare TDE data (with PCA) using the Data class. See the `Data Preparation tutorial <https://osf.io/dx4k2>`_ for further details. Let's prepare TDE-PCA data with 15 embeddings and 80 PCA components.

training_data.prepare(n_embeddings=15, n_pca_components=80)

#%%
# If we now print the Data object, we see the number of channels is the number of PCA components:

print(training_data)

#%%
# Notice, the total number of samples (`n_samples`) has also changed. This is due to the time-delay embedding. We lose `n_embeddings // 2` time points from either end of the time series for each subject. I.e. each subject has lost the first 7 and last 7 data points. We can see we have lost 726000 - 725860 = 140 = 10 * 14 data points. We can double check this by printing the shape of the prepared data for each subject and the shape of the original data.

# Get the original data time series
original_data = training_data.raw_data

# Get the prepared data time series
prepared_data = training_data.time_series()

for i in range(training_data.n_subjects):
    print(original_data[i].shape, prepared_data[i].shape)

#%%
# We can see each subject has lost 14 data points. You will always lose a total of `n_embeddings - 1` data points when performing time-delay embedding.
# 
# The Config object
# *****************
# 
# Now we have prepared the data, let's build a model to train. To do this we first need to specify the `Config object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Config>`_ for DyNeMo. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
# 
# The important hyperparameters to specify are:
#
# - `n_modes`, the number of modes. Unfortunately, this is a hyperparameters that must be pre-specified. We advise starting with something between 6-12 and making sure any results based on the DyNeMo are not critically sensitive to the choice for `n_modes`. In this tutorial, we'll use 6 modes.
# - `sequence_length`. This is a continuous segment that represents one training example. DyNeMo utilises recurrent neural networks (RNNs) which need to be evaluated sequentially. This makes training with very long sequences slow. We advise a sequence length between 50-400. We find a sequence length of 100 or 200 works well.
# - Inference RNN parameters: `inference_n_units` and `inference_normalization`. This is the number of units/neurons and the normalization used in the RNN used to infer the mixing coefficients respectively. (The inference RNN outputs the posterior distribution). The values below should work well in most cases.
# - Model RNN: `model_n_units` and `model_normalization`. Same as above but for the model RNN, which is part of the generative model. (The model RNN outputs the prior distribution). The values given below should work well for most cases.
# - Softmax function parameters: `learn_alpha_temperature` and `initial_alpha_temperature`. The softmax transformation is used to ensure the mode mixing coefficients in DyNeMo are positive and sum to one. This transformation has a 'temperature' parameter that controls how much mixing occurs. We can learn this temperature from the data by specifying `learn_alpha_temperature=True`. We recommend doing this with `initial_alpha_temperature=1.0`.
# - `learn_means` and `learn_covariances`. Typically, if we train on amplitude envelope data we set `learn_means` and `learn_covariances` to `True`, whereas if you're training on time-delay embedded/PCA data, then we only learn the covariances, i.e. we set `learn_means=False`.
# - KL annealing parameters: `do_kl_annealing`, `kl_annealing_curve`, `kl_annealing_sharpness`, `n_kl_annealing_epochs`. When we perform variational Bayesian inference we minimise the variational free energy, which consists of two terms: the log-liklihood term and KL divergence term. We slowly turn on the KL divergence term in the loss function as training progresses. This process is known as **KL annealing**. These parameters control how quickly we turn on the KL term. It is recommended you use `do_kl_annealing=True` if you don't use a pretrained model RNN (which will be the majority of cases). The `kl_annealing_curve` and `kl_annealing_sharpness` values given below will generally work well for most cases. We find using `n_kl_annealing_epochs=n_epochs//2` works well for the duration of KL annealing.
# - `batch_size`. You want a large batch size for fast training. However, you will find that holding large batches requires a lot of memory. Therefore, you should pick the largest values that you can hold in memory. We find a batch size of 8-64 works well for neuroimaging data.
# - `learning_rate`. On large datasets, we find a lower learning rate leads to a lower final loss. We recommend a value between 1e-2 and 1e-4. We advise training a few values and seeing which produces the lowest loss value.
# - `n_epochs`, the number of epochs. This is the number of times you loop through the data. We recommend a value between 50-100 for small datasets (<50 subjects). For really large datasets (100s of subjects) you could train a model with 10 epochs. You can look at the loss as a function of epochs (see below) to see when the model has stopped improving. You can use this as an indicator for when you can stop training.
# 
# In general, you can use the final loss value (lower is better) to select a good set of hyperparameters. Note, we want to compare the full loss function (after the KL term has fully turned on), so you should only use the loss after `n_kl_annealing_epochs` of training have been performed.

from osl_dynamics.models.dynemo import Config

# Create a config object
config = Config(
    n_modes=6,
    n_channels=80,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    n_kl_annealing_epochs=10,
    batch_size=32,
    learning_rate=0.01,
    n_epochs=10,  # for the purposes of this tutorial we'll just train for a short period
)

#%%
# Building the model
# ******************
# 
# With the Config object, we can build a model.

from osl_dynamics.models.dynemo import Model

# Initiate a Model class and print a summary
model = Model(config)
model.summary()

#%%
# Training the model
# ******************
# 
# Note, this step can be time consuming. Training this model on 10 subjects an M1 Macbook Air takes ~2 minute per epoch, which leads to a total training time of around 20 minutes.
# 
# **Initialization**
# 
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the mode means/covariances helps find a good solution. The `dynemo.Model <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Model>`_ class has the `random_subset_initialization` method that can be used for this. This method train the model for a short period on a small random subset of the data. Let's use this method to initialize the model.

init_history = model.random_subset_initialization(training_data, n_epochs=1, n_init=3, take=0.2)

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
# 
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
# 
# We're interested in the inferred mixing coefficients and mode covariances.
# 
# Mixing coefficients
# *******************
# 
# DyNeMo performs Bayesian inference on the mixing coefficients, i.e. it learns the probability of the mixing ratios at each time point. We can get the most likely mixing coefficients using the `get_alpha` method of the model.

alpha = model.get_alpha(training_data)

#%%
# `alpha` is a list of numpy arrays that contains a `(n_samples, n_modes)` time series, which is the mixing coefficients at each time point. Each item of the list corresponds to a subject. We can further understand the `alpha` list by printing the shape of its items.

for a in alpha:
    print(a.shape)

#%%
# Notice, the number of samples in the state probability time series for each subject does not match the number of samples in the prepared data for each subject. Printing them side by side, we have:

for a, x in zip(alpha, prepared_data):
    print(a.shape, x.shape)

#%%
# This is due to the training process which first segments the data for each subject into sequences of length `sequence_length` and then groups them into groups of size `batch_size`. If there are extra data points in a subject's time series that do not fit into a sequence, they are dropped. Looking at the first subject, we see the mixing coefficient time series (`alpha[0]`) has 74400 samples, but the prepared data has 74486 samples, this means the first subject was segmented into `74486 // sequence_length = 74486 // 100 = 744` sequences, which will give `744 * sequence_length = 744 * 100 = 74400` samples in the mixing coefficient time course.
# 
# We often want to align the inferred mixing coefficients to the original (unprepared) data. This can be done using the `get_training_time_series` method of a model. Note, this method is defined in the `model base class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/mod_base/index.html>`_, which is inherited by the `dynemo.Model` class. Let's use this function to get the original source reconstructed data that each time point in the state probability time course corresponds to.

data = model.get_training_time_series(training_data, prepared=False)

#%%
# `data` is a list of numpy arrays, one for each subject. Now if print the shape of the mixing coefficient time course and training data, we'll see they match.

for a, x in zip(alpha, data):
    print(a.shape, x.shape)

#%%
# We don't want to have to keep loading the model to get the inferred mixing coefficient, instead let's just save the inferred mixing coefficients.

import pickle

# Save the python lists as a pickle files
os.makedirs("results/data", exist_ok=True)
pickle.dump(alpha, open("results/data/alpha.pkl", "wb"))

#%%
# Mode covariances
# ****************
# 
# DyNeMo learns point estimates for the mode covariances. We can get the inferred covariances using the `get_covariances` method of the model.

import numpy as np

# Get the inferred covariances
covs = model.get_covariances()

# Save
np.save("results/data/covs.npy", covs)

#%%
# Note, if we were learning the means we could get the inferred means and covariances using the `get_means_covariances` method.
# 
# Calculating Mode Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Using the inferred mixing coefficients and training data (unprepared) we can calculate the spectral properties of each mode.
# 
# Power spectra and coherences
# ****************************
# 
# We want to calculate the power spectrum and coherence of each mode. A linear regression approach where we regress the spectrogram (time-varying spectra) with the inferred mixing coefficients. We do this with the `analysis.spectral.regression_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.regression_spectra>`_ function in osl-dynamics. Let's first run this function, then we'll discuss its output. The arguments we need to pass to this function are:
#
# - `data`. This is the source reconstructed data aligned to the mixing coefficients.
# - `alpha`. This is the mixing coefficient time series.
# - `sampling_frequency` in Hz.
# - `frequency_range`. This is the frequency range we're interested in.
# - `window_length`. This is the length of the data segment (window) we use to calculate the spectrogram.
# - `step_size`. This is the number of samples we slide the window along the data when calculating the spectrogram.
# - `n_sub_windows`. To calculate the coherence we need to split the window into sub-windows and average the cross specrta. This is the number of sub-windows.
# - `return_coef_int`. We split a linear regression to the spectrogram. We may want to receive the regression coefficients and intercept separately.

from osl_dynamics.analysis import spectral

# Calculate regression spectra for each state and subject (will take a few minutes)
f, psd, coh = spectral.regression_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[0, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
)

#%%
# Note, there is a `n_jobs` argument that can be used to calculate the regression spectra for each subject in parallel.
# 
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.

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
# 
# - We have shown how to prepare time-delay embedded/PCA data for training DyNeMo.
# - We have trained and saved DyNeMo and its inferred parameters (mixing coefficients, covariances).
# - We have calculated the spectral properties of each mode.
