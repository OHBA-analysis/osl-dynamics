"""
HMM: Training
=============

This tutorial covers how to train a Hidden Markov Model (HMM). We will use MEG data in this tutorial, however, this can easily be substituted with fMRI data.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed and prepared. This dataset is:
#
# - From 51 subjects.
# - Parcellated to 52 regions of interest (ROI). The parcellation file used was `Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# - Prepared using 15 time-delay embeddings and 120 PCA components.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.


import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 1.7 GB)
get_data("notts_mrc_meguk_glasser_prepared")

#%%
# Load the data
# *************
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.


from osl_dynamics.data import Data

data = Data("notts_mrc_meguk_glasser_prepared")
print(data)

#%%
# Fitting an HMM
# ^^^^^^^^^^^^^^
#
# The Config object
# *****************
# Let's build a model to train. To do this we first need to specify the `Config object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Config>`_ for the HMM. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
#
# An important hyperparameters to specify is `n_states`, which the number of states. We advise starting with something between 6-14 and making sure any results based on the HMM are not critically sensitive to the choice for `n_states`. In this tutorial, we'll use 8 states.
#
# The `sequence_length` and `batch_size` can be chosen to ensure the model fits into memory. 


from osl_dynamics.models.hmm import Config

config = Config(
    n_states=8,
    n_channels=data.n_channels,
    sequence_length=1000,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=20,
)

#%%
# Note, in fMRI we have much shorter time series. This means we need to use a shorter sequence length and larger batch size, for example::
#
#     config = Config(
#         n_states=8,
#         n_channels=data.n_channels,
#         sequence_length=50,
#         learn_means=False,
#         learn_covariances=True,
#         batch_size=256,
#         learning_rate=0.01,
#         n_epochs=20,
#     )
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
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the state means/covariances helps find a good explanation. The `hmm.Model <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Model>`_ class has a few helpful methods for initialization. When training on real data, we recommend using the `random_state_time_course_initialization`, let's do this. Usually 3 initializations is enough and you only need to train for a short period, we will use a single epoch.


init_history = model.random_state_time_course_initialization(data, n_epochs=1, n_init=3)

#%%
# The `init_history` variable is `dict` that contains the training history (`rho`, `lr`, `loss`) for the best initialization.
#
# **Full training**
#
# Now, we have found a good initialization, let's do the full training of the model. We do this using the `fit` method.


history = model.fit(data)

#%%
# The `history` variable contains the training history of the `fit` method.
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
#     model = load("results/model")
