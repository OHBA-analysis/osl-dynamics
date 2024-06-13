"""
HMM/DyNeMo: Get Inferred Parameters
===================================

In this tutorial we will get the inferred parameters from a dynamic network model (HMM/DyNeMo).
"""

#%%
# Load Trained Model
# ^^^^^^^^^^^^^^^^^^
# First we need to load the trained model.


from osl_dynamics.models import load

model = load("results/model")

#%%
# Getting the Inferred Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Latent variables
# ****************
# Both the HMM and DyNeMo refer to the latent variable as `alpha`. In the HMM, `alpha` corresponds to the state probability time courses, and in DyNeMo, `alpha` corresponds to the mixing coefficient time courses. We can get these using the `get_alpha` method by passing the (prepared) training data.
#
# Let's first load the training data.


from osl_dynamics.data import Data

data = Data("prepared_data")

#%%
# Now we can  get the `alpha` for each subject.


alpha = model.get_alpha(data)

#%%
# `alpha` is a list of numpy arrays, one for each subject.

import os
import pickle

os.makedirs("results/inf_params", exist_ok=True)
pickle.dump(alpha, open("results/inf_params/alpha.pkl", "wb"))

#%%
# Observation model
# *****************
# We can get the inferred state/mode means and covariances with the `get_means_covariances` method. Note, if we passed `learn_means=False` in the config, the means will be zero.


import numpy as np

# Get the inferred means and covariances
means, covs = model.get_means_covariances()

# Save
np.save("results/inf_params/means.npy", means)
np.save("results/inf_params/covs.npy", covs)

