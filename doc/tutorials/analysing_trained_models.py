"""
Analysing Trained Models
========================
This is a tutorial that covers some basic options for analysing a trained model. In this tutorial we will look at the DyNeMo model.

"""

#%%
# We start by importing the necessary packages.

from osl_dynamics import data, inference
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.utils import plotting

#%%
# Loading a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# To load a model we must rebuild it using the same Config as before. In this tutorial we will rebuild the model trained in the 'Training Models' tutorial.

config = Config(
    n_modes=5,
    n_channels=11,
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
# We rebuild the model and load the trained model weights:

model = Model(config)
model.load_weights("trained_model/weights")

#%%
# Evaluating a Model
# ^^^^^^^^^^^^^^^^^^
#
# The first thing we can do to evaluate a model is to evaluate the loss (variational free energy for DyNeMo) with the training dataset.
# For this we need the training data. Let's load the same training data as the 'Training Models' tutorial:

training_data = data.Data("X.npy")
training_data.prepare()

#%%
# To evaluate our model let's create a TensorFlow dataset without shuffling the sequences and batches to maintain the order of the time series.
# The loss is evaluated by averaging the batches across a dataset. Therefore, it should be the same if you use a shuffled or non-shuffled dataset.

prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

#%%
# NOTE: the `dataset` method has a `concatenate` argument. If `concatenate=False` the `dataset` method will return a list of TensorFlow datasets.
# This is useful if you want to make subject-specific predictions.

#%%
# DyNeMo has a method to evaluate the variational free energy:

free_energy = model.free_energy(prediction_dataset)
print("Free energy:", free_energy)

#%%
# We are often interested in interpreting latent variables. In DyNeMo, these are the mode mixing coefficients, alpha, and mode means and covariances.
#
# The DyNeMo model has methods to get the inferred parameters:

alpha = model.get_alpha(prediction_dataset)
means, covs = model.get_means_covariances()

#%%
# osl-dynamics has many built in functions to summarise the inferred alphas. Most of these are found in the `osl_dynamics.inference.modes` subpackage.
# For example, we can calculate the fractional occupancy of each mode with:

fo = inference.modes.fractional_occupancies(alpha)
print("Fractional occupancies:", fo)

#%%
# Plotting
# ^^^^^^^^
#
# osl-dynamics has many built in functions for plotting. To plot the inferred alphas, we can use

plotting.plot_alpha(alpha, n_samples=2000)

#%%
# NOTE: all functions in `osl_dynamics.utils.plotting` have a `filename` argument where you can pass a string to save an image file instead of opening it interactivately.
