"""
HMM: Resting-State Network Analysis (Part 1)
============================================
 
In this tutorial we will perform dynamic network analysis on source space MEG data using a Hidden Markov Model (HMM). We will focus on resting-state data. This tutorial covers:

1. Getting the Data
2. Fitting an HMM

The input to this script is:

- A set of time series (one for each subject you have). In this tutorial we will download some example data.

The output of this script is:

- A trained HMM.
- The inferred parameters.

Note, this webpage does not contain the output of each cell. We advise downloading the notebook and working through it locally on your machine. The expected output of this script can be found `here <https://osf.io/m43ae>`_.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# 
# We will use eyes open resting-state data that has already been source reconstructed. We call this the 'Nottingham dataset'. This dataset is:
#
# - From 10 subjects.
# - Parcellated to 42 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# 
# Download the dataset
# ********************
# 
# We will download data hosted on OSF.

import os

def get_notts_data():
    """Downloads the Nottingham dataset from OSF."""
    if os.path.exists("notts_dataset"):
        return "notts_dataset already downloaded. Skipping.."
    os.system("osf -p zxb6c fetch Dynamics/notts_dataset.zip")
    os.system("unzip -o notts_dataset.zip -d notts_dataset")
    os.remove("notts_dataset.zip")
    return "Data downloaded to: notts_dataset"

# Download the dataset (it is 113 MB)
get_notts_data()

# List the contents of the downloaded directory containing the dataset
get_ipython().system('ls notts_dataset')

#%%
# Load the data
# *************
# 
# We now load the data into osl-dynamics using the Data object. This is a python class which has a lot of useful methods that can be used to modify the data. See the `API reference guide <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ for further details.

from osl_dynamics.data import Data

training_data = Data("notts_dataset")

# Display some summary information
print(training_data)

#%%
# Note, when we load data using the Data object, it creates a `/tmp` directory which is used for storing temporary data. This directory can be safely deleted after you run your script. You can specify the name of the temporary directory by pass the `store_dir="..."` argument to the Data object.
# 
# For static analysis we just need the time series for the parcellated data. We can access this using the `Data.time_series` method. This returns a list of numpy arrays. Each numpy array is a `(n_samples, n_channels)` time series for each subject.
#
# Fitting an HMM
# ^^^^^^^^^^^^^^
# 
# Now we have loaded the data, let's see how to fit an HMM to it. In this tutorial, we want to study the spectral properties of dynamic networks. To do this we will use an approach known as time-delay embedding (TDE) to capture the spectral properties of each network in the model. This is commonly referred to as fitting a TDE-HMM.
# 
# Preparing the data (time-delay embedding)
# *****************************************
# 
# Time-delay embedding involves adding extra channels to our training data we contain time shifted versions of the original channels. By doing this we introduce additional rows and columns into the covariance matrix of our data. These additional elements capture spectral (i.e. frequency-specific) characteristics of our data.
# 
# The `Data object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ has a method for preparing the data called `prepare`. To time-delay embed the data, we just need to pass the `n_embeddings` argument. When we perform time-delay embedding we add a lot of additional channels, this generally leads to the training data being too large to hold in memory, therefore an additional step we often take is Principal Component Analysis (PCA) to remove the number of channels. We can do this by pass a `n_pca_components` argument to the `prepare` method. Typically with MEG data, we use `n_embeddings=15`, which corresponds to adding channels with -7, -6, ..., 0, 6, 7 lags, and `n_pca_components=80`.

training_data.prepare(n_embeddings=15, n_pca_components=80)

#%%
# Note, the `prepare` method always performs standardization (z-tranform) as the final step. If we now print the Data object, we see the number of channels is the number of PCA components:

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
# Now we have prepared the data, let's build a model to train. To do this we first need to specify the `Config object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Config>`_ for the HMM. This is a class that acts as a container for all hyperparameters of a model. The API reference guide lists all the arguments for a Config object. There are a lot of arguments that can be passed to this class, however, a lot of them have good default values you don't need to change.
# 
# The important hyperparameters to specify are:
#
# - `n_states`, the number of states. Unfortunately, this is a hyperparameters that must be pre-specified. We advise starting with something between 6-14 and making sure any results based on the HMM are not critically sensitive to the choice for `n_states`. In this tutorial, we'll use 8 states.
# - `sequence_length` and `batch_size`. You want a large sequence length/batch size for fast training. However, you will find that holding large batches of long sequences requires a lot of memory. Therefore, you should pick the largest values that you can hold in memory. We find a sequence length of 1000-2000 and batch size of 8-64 works well for real neuroimaging data.
# - `learn_means` and `learn_covariances`. Typically, if we train on amplitude envelope data we set `learn_means` and `learn_covariances` to `True`, whereas if you're training on time-delay embedded/PCA data, then we only learn the covariances, i.e. we set `learn_means=False`.
# - `learning_rate`. On large datasets, we find a lower learning rate leads to a lower final loss. We recommend a value between 1e-2 - 1e-4. We advice training a few values and seeing which produces the lowest loss value.
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
# 
# With the Config object, we can build a model.

from osl_dynamics.models.hmm import Model

# Initiate a Model class and print a summary
model = Model(config)
model.summary()

#%%
# Training the model (could take over 20 minutes)
# ***********************************************
# 
# Note, training this model on an M1 Macbook Air takes ~1 minute per epoch.
# 
# When training a model it often helps to start with a good initialization. In particular, starting with a good initial value for the state means/covariances helps find a good explanation. The `hmm.Model <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html#osl_dynamics.models.hmm.Model>`_ class has a few helpful methods for initialization. When training on real data, we recommend using the `random_state_time_course_initialization`, let's do this. Usually 3 initializations is enough and you only need to train for a short period, we will use a single epoch.

init_history = model.random_state_time_course_initialization(training_data, n_epochs=1, n_init=3)

#%%
# The `init_history` variable is `dict` that contains the training history (`rho`, `lr`, `loss`) for the best initialization.
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

model.save("trained_model")

#%%
# This will automatically create a directory containing the trained model weights and config settings used. Note, should we wish to load the trained model we can use::
#
#     from osl_dynamics.models import load
# 
#     # Load the trained model
#     model = load("trained_model")
# 
# Getting the inferred parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
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
# We often want to align the inferred state probability to the original (unprepared) data. This can be done using the `get_training_time_series` method of a model. Note, this method is defined in the model [base class](https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/mod_base/index.html), which is inherited by the `hmm.Model` class. Let's use this function to get the original source reconstructed data that each time point in the state probability time course corresponds to.

data = model.get_training_time_series(training_data, prepared=False)

#%%
# `data` is a list of numpy arrays, one for each subject. Now if print the shape of the state probability time course and training data, we'll see they match.

for a, x in zip(alpha, data):
    print(a.shape, x.shape)

#%%
# We don't want to have to keep loading the model to get the inferred state probabilities, instead let's just save the inferred state probabilties. It's also helpful to save the training data aligned to the state probabilties, let's do this.

import pickle

# Save the python lists as a pickle files
pickle.dump(alpha, open("trained_model/alpha.pkl", "wb"))
pickle.dump(data, open("trained_model/data.pkl", "wb"))

#%%
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to prepare time-delay embedded/PCA data for training an HMM.
# - We have trained and saved an HMM and its inferred parameters (state probabilities).
