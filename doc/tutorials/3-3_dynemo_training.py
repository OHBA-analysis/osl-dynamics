"""
DyNeMo: Training
================

This tutorial covers how to train a DyNeMo model. We will use MEG data in this tutorial, however, this can easily be substituted with fMRI data.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed and prepared. This dataset is:
#
# - Parcellated to 38 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# - Prepared using 15 time-delay embeddings and 80 PCA components.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.
#
# .. code-block:: python
#
#     import os
#
#     def get_data(name, rename):
#         if rename is None:
#             rename = name
#         if os.path.exists(rename):
#             return f"{name} already downloaded. Skipping.."
#         os.system(f"osf -p by2tc fetch data/{name}.zip")
#         os.makedirs(rename, exist_ok=True)
#         os.system(f"unzip -o {name}.zip -d {rename}")
#         os.remove(f"{name}.zip")
#         return f"Data downloaded to: {rename}"
#
#     # Download the dataset (approximately 21 MB)
#     get_data("notts_mrc_meguk_giles_prepared_1_subject", rename="prepared_data")

#%%
# Load the data
# *************
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.
#
# .. code-block:: python
#
#     from osl_dynamics.data import Data
#
#     data = Data("prepared_data")
#     print(data)

#%%
# Note, we can pass `use_tfrecord=True` when creating the Data object if we are training on large datasets and run into an out of memory error.
#
# Fitting DyNeMo
# ^^^^^^^^^^^^^^
#
# The Config object
# *****************
# Now we have prepared the data, let's build a model to train. To do this we first need to specify the `Config object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Config>`_ for DyNeMo. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
#
# The important hyperparameters to specify are:
#
# - `n_modes`, the number of modes. Unfortunately, this is a hyperparameters that must be pre-specified. We advise starting with something between 6-12 and making sure any results based on the DyNeMo are not critically sensitive to the choice for `n_modes`. In this tutorial, we'll use 6 modes.
# - `sequence_length`. This is a continuous segment that represents one training example. DyNeMo utilises recurrent neural networks (RNNs) which need to be evaluated sequentially. This makes training with very long sequences slow. We advise a sequence length of 200 or less. 100 is often a good choice.
# - Inference RNN parameters: `inference_n_units` and `inference_normalization`. This is the number of units/neurons and the normalization used in the RNN used to infer the mixing coefficients respectively. (The inference RNN outputs the posterior distribution). The values below should work well in most cases.
# - Model RNN: `model_n_units` and `model_normalization`. Same as above but for the model RNN, which is part of the generative model. (The model RNN outputs the prior distribution). The values given below should work well for most cases.
# - Softmax function parameters: `learn_alpha_temperature` and `initial_alpha_temperature`. The softmax transformation is used to ensure the mode mixing coefficients in DyNeMo are positive and sum to one. This transformation has a 'temperature' parameter that controls how much mixing occurs. We can learn this temperature from the data by specifying `learn_alpha_temperature=True`. We recommend doing this with `initial_alpha_temperature=1.0`.
# - `learn_means` and `learn_covariances`. Typically, if we train on amplitude envelope data we set `learn_means` and `learn_covariances` to `True`, whereas if you're training on time-delay embedded/PCA data, then we only learn the covariances, i.e. we set `learn_means=False`.
# - KL annealing parameters: `do_kl_annealing`, `kl_annealing_curve`, `kl_annealing_sharpness`, `n_kl_annealing_epochs`. When we perform variational Bayesian inference we minimise the variational free energy, which consists of two terms: the log-liklihood term and KL divergence term. We slowly turn on the KL divergence term in the loss function as training progresses. This process is known as **KL annealing**. These parameters control how quickly we turn on the KL term. It is recommended you use `do_kl_annealing=True` if you don't use a pretrained model RNN (which will be the majority of cases). The `kl_annealing_curve` and `kl_annealing_sharpness` values given below will generally work well for most cases. We find using `n_kl_annealing_epochs=n_epochs//2` works well for the duration of KL annealing.
# - `batch_size`. You want a large batch size for fast training. However, you will find that holding large batches requires a lot of memory. Therefore, you should pick the largest values that you can hold in memory. We find a batch size of 8-64 works well for neuroimaging data.
# - `learning_rate`. On large datasets, we find a lower learning rate leads to a lower final loss. We recommend a value between 1e-2 and 1e-4. We advise training a few values and seeing which produces the lowest loss value.
# - `n_epochs`, the number of epochs. This is the number of times you loop through the data. We recommend a value between ~50 for small datasets (<50 subjects). For large datasets (100s of subjects) you could train a model with 10 epochs. You can look at the loss as a function of epochs (see below) to see when the model has stopped improving. You can use this as an indicator for when you can stop training.
#
# In general, you can use the final loss value (lower is better) to select a good set of hyperparameters. Note, we want to compare the full loss function (after the KL term has fully turned on), so you should only use the loss after `n_kl_annealing_epochs` of training have been performed.

from osl_dynamics.models.dynemo import Config

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
    n_epochs=20,
)

#%%
# Building the model
# ******************
# With the Config object, we can build a model.

from osl_dynamics.models.dynemo import Model

model = Model(config)
model.summary()

#%%
# Training the model
# ******************
# Note, this step can be time consuming.
#
# **Initialization**
#
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the mode means/covariances helps find a good solution. The `dynemo.Model <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Model>`_ class has the `random_subset_initialization` method that can be used for this. This method train the model for a short period on a small random subset of the data. Let's use this method to initialize the model.
#
# .. code-block:: python
#
#     init_history = model.random_subset_initialization(data, n_epochs=2, n_init=5, take=0.25)

#%%
# The `init_history` variable is `dict` that contains the training history (`rho`, `lr`, `loss`) for the best initialization.
#
# **Full training**
#
# Now, we have found a good initialization, let's do the full training of the model. We do this using the `fit` method.
#
# .. code-block:: python
#
#     history = model.fit(data)

#%%
# The `history` variable contains the training history of the `fit` method.
#
# Saving a trained model
# **********************
# As we have just seen, training a model can be time consuming. Therefore, it is often useful to save a trained model so we can load it later. We can do this with the `save` method.
#
# .. code-block:: python
#
#     model.save("results/model")

#%%
# This will automatically create a directory containing the trained model weights and config settings used. Note, should we wish to load the trained model we can use:
#
# .. code-block:: python
#
#     from osl_dynamics.models import load
#
#     model = load("results/model")

#%%
# It's also useful to save the variational free energy to compare different runs.
#
# .. code-block:: python
#
#     import pickle
#
#     free_energy = model.free_energy(data)
#     history["free_energy"] = free_energy
#     pickle.dump(history, open("results/model/history.pkl", "wb"))
