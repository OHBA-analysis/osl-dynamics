"""Get the inferred parameters for a dataset given by a trained model.

This script will work for any model in osl-dynamics.

For DyNeMo the inferred parameters are the mode mixing coefficients (alpha_jt)
and the inferred covariances.

For the HMM the inferred parameters are the state time course and covariances.
"""

import numpy as np
import pickle

from osl_dynamics.data import Data
from osl_dynamics.models import load

# Directory containing the prepared data
data_dir = "data"

# Directory containing the model
model_dir = "model"

# Load the model
model = load(model_dir)

# Load the data used to train the model
data = Data(data_dir)

# Get the inferred parameters
#
# For DyNeMo this will give the mixing coefficients,
# for the HMM it will give the state time course.
#
# alp will be a list of subject-specific numpy arrays in (time x n_modes)
# format. Note, alp will have slightly fewer samples than the original
# (unprepared) data if you applied time-delay embeddings. n_embeddings // 2
# data points will be removed from each end of the time series.
alp = model.get_alpha(training_data)

# Get the inferred observation model parameters (i.e. covariances)
#
# covs will be a (n_modes, n_channels, n_channels) numpy array
covs = model.get_covariances()

# Save the inferred parameters (this will save them to the same
# directory as the model)
pickle.dump(alp, open(model_dir + "/alp.pkl"))
np.save(model_dir + "/covs.npy", covs)
