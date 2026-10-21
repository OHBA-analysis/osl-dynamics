"""
HMM: Training
=============

This tutorial covers how to train an HMM. We will use MEG data in this tutorial, however, this can easily be substituted with fMRI data.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed and prepared. This dataset is:
#
# - Parcellated to 38 regions of interest (ROI). The parcellation file used was `atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz`.
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
#         os.system(f"osf -p by2tc fetch data/{name}.zip")
#         os.makedirs(rename, exist_ok=True)
#         os.system(f"unzip -o {name}.zip -d {rename}")
#         os.remove(f"{name}.zip")
#         return f"Data downloaded to: {rename}"
#
#     # Download the dataset (approximately 21 MB)
#     get_data("notts_meguk_giles_prepared_1_subject", rename="prepared_data")

#%%
# Load the data
# *************
# We now load the data into osl-dynamics using the Data class. See the :doc:`Loading Data tutorial <../tutorials_build/1-1_data_loading>` for further details.
#
# .. code-block:: python
#
#     from osl_dynamics.data import Data
#
#     data = Data("prepared_data")
#     print(data)

#%%
# If you run into an out of memory error when training, see the `Training on large datasets` section below.
#
# Fitting an HMM
# ^^^^^^^^^^^^^^
#
# The Config object
# *****************
# First need to specify the :class:`Config object <osl_dynamics.models.hmm.Config>` for the HMM. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
#
# An important hyperparameters to specify is `n_states`, which the number of states. We advise starting with something between 6-14 and making sure any results based on the HMM are not critically sensitive to the choice for `n_states`. In this tutorial, we’ll use 6 states.
#
# The `sequence_length` and `batch_size` can be chosen to ensure the model fits into memory.

from osl_dynamics.models.hmm import Config

# Create a config object
config = Config(
    n_states=6,
    n_channels=80,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=256,
    learning_rate=0.01,
    n_epochs=20,
)

#%%
# **Note**: for fMRI data the `sequence_length` used is typically much shorter, normally `sequence_length=50`.
#
# Building the model
# ******************
# With the Config object, we can build a model.

from osl_dynamics.models.hmm import Model

model = Model(config)
model.summary()

#%%
# Training the model
# ******************
# Note, this step can be time consuming.
#
# **Initialization**
#
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the state means/covariances helps find a good explanation. The :class:`hmm.Model <osl_dynamics.models.hmm.Model>` class has a few helpful methods for initialization. When training on real data, we recommend using the `random_state_time_course_initialization`, let's do this. Usually 3 initializations is enough and you only need to train for a short period, we will use a single epoch.
#
# .. code-block:: python
#
#     init_history = model.random_state_time_course_initialization(data, n_epochs=1, n_init=3)

#%%
# If you are training on a particularly large dataset, initializing on the full dataset can be slow. In this case, you can pass the `take` argument to initialize on a random subset of the data. This is the fraction of the dataset to use, e.g. `take=0.25` will use a quarter of the data for each initialization.
#
# .. code-block:: python
#
#     init_history = model.random_state_time_course_initialization(data, n_epochs=1, n_init=3, take=0.25)

#%%
# The `init_history` variable is `dict` that contains the training history (`rho`, `learning_rate`, `loss`) for the best initialization.
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

#%%
# Training on large datasets
# **************************
# By default the Data object holds the data in memory and cuts sequences out of it on the fly, so the only extra memory used during training is the batches in flight. If your dataset does not fit in memory, save the prepared data as a TFRecord dataset, which is streamed from disk during training. This also means the data is only prepared once if you train many models on it. The `sequence_length` must match the one in the model's Config.
#
# .. code-block:: python
#
#     data.save_tfrecord_dataset("tfrecords", sequence_length=200)
#
# Then in your training script, load the dataset and pass it to the model in place of the Data object.
#
# .. code-block:: python
#
#     from osl_dynamics.data import load_tfrecord_dataset
#
#     dataset = load_tfrecord_dataset("tfrecords", batch_size=config.batch_size)
#     model.random_state_time_course_initialization(dataset, n_epochs=1, n_init=3)
#     model.fit(dataset)
#
# This dataset is shuffled for training. To get inferred parameters in time order (e.g. with `get_alpha`), load an unshuffled dataset with one dataset per session.
#
# .. code-block:: python
#
#     dataset = load_tfrecord_dataset("tfrecords", batch_size=config.batch_size, shuffle=False, concatenate=False)
#     alpha = model.get_alpha(dataset)
#
# The `buffer_size` argument of `load_tfrecord_dataset` sets how many sequences are held in memory for shuffling (default 4000); reduce it if you still run out of memory. If you run out of GPU memory rather than RAM, reduce the `batch_size` or `sequence_length` in the Config.
#
# Alternatively, the Data object can keep the data on disk as memory maps (`Data(..., load_memmaps=True)`) and read sequences from disk as they are needed, which avoids the extra step of writing TFRecords.

#%%
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
