"""
Analysing Trained Models
========================
This is a tutorial that covers some basic options for analysing a trained model. In this tutorial we will look at the DyNeMo model.

"""

#%%
# We start by importing the necessary packages.

from osl_dynamics.analysis import modes
from osl_dynamics.data import Data
from osl_dynamics.models import load
from osl_dynamics.utils import plotting

#%%
# Loading a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# To load a model we can use:

model = load("trained_model")

#%%
# Evaluating a Model
# ^^^^^^^^^^^^^^^^^^
#
# To evaluate a model we can use the loss (variational free energy). To calculate this we need to load the training dataset. Let's load the same training data as the 'Training Models' tutorial:

training_data = Data("X.npy")
training_data.prepare()

#%%
# DyNeMo has a method to evaluate the variational free energy:

free_energy = model.free_energy(training_data)
print("Free energy:", free_energy)

#%%
# We are often interested in interpreting latent variables. In DyNeMo, these are the mode mixing coefficients, alpha, and mode means and covariances. The DyNeMo model has methods to get the inferred parameters:

alpha = model.get_alpha(training_data)
means, covs = model.get_means_covariances()

#%%
# osl-dynamics has many built in functions to summarise the inferred alphas. Most of these are found in the `osl_dynamics.analysis.modes` subpackage.
# For example, we can calculate the fractional occupancy of each mode with:

fo = modes.fractional_occupancies(alpha)
print("Fractional occupancies:", fo)

#%%
# Plotting
# ^^^^^^^^
#
# osl-dynamics has many built in functions for plotting. To plot the inferred alphas, we can use

plotting.plot_alpha(alpha, n_samples=2000)

#%%
# Note, all functions in osl_dynamics.utils.plotting have a `filename` argument where you can pass a string to save an image file instead of opening it interactivately.
