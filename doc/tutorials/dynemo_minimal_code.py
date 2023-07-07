"""
DyNeMo: Minimal Code
====================

This tutorial covers example use of DyNeMo (Dynamic Network Modes) with the minimum amount of code possible.
"""

#%%
# Importing Packages
# ^^^^^^^^^^^^^^^^^^
# We start by importing the necessary packages:

from osl_dynamics.data import Data
from osl_dynamics.models import load
from osl_dynamics.models.dynemo import Config, Model

#%%
# Loading Data
# ^^^^^^^^^^^^
# Data can be loaded from a file in osl-dynamics with the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details. The Data class can accept:
#
# - A string containing the path to a `.npy`, `.mat` or `.fif` file.
# - A list of strings containing paths to `.npy`, `.mat` or `.fif` files.
# - A path to a directory containing `.npy` files.
# - A numpy array can also be passed directly to Data.
#
# NOTE:
#
# - If a `.mat` file is passed, it must have a field called "X" that contains the data or you need to pass the `data_field` argument to the Data class.
# - The data must be in the format (n_samples, n_channels), if you have the data in (n_channels, n_samples) format you can pass `time_axis_first=False`.

training_data = Data([f"subject{i}.npy" for i in range(10)])

#%%
# You can pass your own data to the Data object, any numpy file containing a 2D array should be fine.
#
# Preparing the Data
# ^^^^^^^^^^^^^^^^^^
# The Data class can be used to apply standard transformations to the data to prepare it for training.
# The `prepare` method can be used for this. See the `Preparing Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_preparation.html>`_ for further details. Time-delay embedded/PCA data can be prepared with:

methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
training_data.prepare(methods)

#%%
# Calling `training_data.time_series()` will now return the prepared data. The original data is still accessible with `training_Data.time_series(prepared=False)`.
#
# Saving and Loading Prepared Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is sometimes useful to save prepared data. This can be done with

training_data.save("prepared_data")

#%%
# This will create a directory called `prepared_data` with `.npy` files containing the prepared data and a pickle file with preparation settings.
# The prepared data can be loaded in another script with:

training_data = Data("prepared_data")

#%%
# The Config Object
# ^^^^^^^^^^^^^^^^^
# Next, we want to train a model on this data. We first need to specify hyperparameters for the model. These are contained in the Config object. The API reference for each model lists the attributes for each model's Config object.
#
# An example Config object for DyNeMo is:

config = Config(
    n_modes=6,
    n_channels=80,
    sequence_length=200,
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
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

#%%
# Important decisions to be made are:
#
# - The number of modes, `n_modes`. For this you should make sure your results are not critically depends on the choice for this.
# - `sequence_length` and `batch_size`. If you find you run out of memory, you can reduce these. Otherwise, increasing the batch size will lead to faster training. We have found a sequence length of 100-400, generally works well.
# - `learn_means` and `learn_covariances`. Typically, we only learn the mean if we're training on amplitude envelope data. If we are training on time-delay embedded/PCA data, we just learn the covariances.
# - `n_kl_annealing_epochs` and `n_epochs`. We found using `n_kl_annealing_epochs = n_epochs // 2` works well. You want to choice `n_epochs` to be enough for the loss to converge during training.
#
# Building a Model
# ^^^^^^^^^^^^^^^^
# We build a model using the Model class and Config object:

model = Model(config)

#%%
# We can treat the model object as a normal TensorFlow Keras Model object, e.g. to view a summary, we can use

model.summary()

#%%
# Training a Model
# ^^^^^^^^^^^^^^^^
# To train the model we use the fit() method:

history = model.fit(training_data)

#%%
# Saving and Loading Trained Models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To save a trained model we can use:

model.save("trained_model")

#%%
# To load a model we can use:

model = load("trained_model")

#%%
# Evaluating a Model
# ^^^^^^^^^^^^^^^^^^
# To evaluate a model we can use the loss (variational free energy). DyNeMo has a method to evaluate the variational free energy:

free_energy = model.free_energy(training_data)
print("Free energy:", free_energy)

#%%
# Getting Inferred Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We are often interested in interpreting latent variables. In DyNeMo, these are the mode mixing coefficients, alpha, and mode means and covariances. The DyNeMo model has methods to get the inferred parameters:

alpha = model.get_alpha(training_data)
means, covs = model.get_means_covariances()
