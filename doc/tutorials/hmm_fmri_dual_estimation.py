"""
HMM: fMRI Dual Estimation
=========================

After training an HMM on fMRI data, we normally want to estimate subject-specific quantities based on the group-level model. We refer to this process as 'dual estimation'. We estimated subject/static-specific means and covariances. In this tutorial, we show how to do this with osl-dynamics.
"""

#%%
# Load trained model
# ^^^^^^^^^^^^^^^^^^
# First we need to load the trained model.


from osl_dynamics.models import load

model = load("results/model")

#%%
# Load data and state probabilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First we need to load the (prepared) training data.


from osl_dynamics.data import Data

data = Data("prepared_data")

#%%
# We also need the state probabilities for each subject.


import pickle

pickle.load(open("results/inf_params/alpha.pkl", "wb"))

#%%
# Dual estimation
# ^^^^^^^^^^^^^^^
# The HMM has a method for dual estimation, which calculates subject/static-specific means/covariances based on the state probabilities.


means, covs = model.dual_estimation(data, alpha=alpha)

#%%
# Note, if we haven't calculated the state probabilities, we can just pass the data::
#
#     means, covs = model.dual_estimation(data)
#
# and it'll calculate the state probabilities for us.
#
# Finally, it's useful to save these to load later.


import os
import numpy as np

os.makedirs("results/dual_estimates", exist_ok=True)
np.save("results/dual_estimates/means.npy", means)
np.save("results/dual_estimates/covs.npy", covs)

