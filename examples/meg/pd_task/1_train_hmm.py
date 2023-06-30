"""Train an HMM on time-delay embedded/PCA data.

"""

import os
import pickle
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

# Directories
model_dir = "results/model"
inf_params_dir = "results/inf_params"

os.makedirs(inf_params_dir, exist_ok=True)

#%% Prepare data

# Load data
training_data = Data([f"data/subject{i}.npy" for i in range(30)], n_jobs=4)

# Perform time-delay embedding and PCA
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
training_data.prepare(methods)

#%% Setup model

config = Config(
    n_states=8,
    n_channels=80,
    sequence_length=2000,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=32,
    learning_rate=1e-3,
    n_epochs=20,
)

model = Model(config)
model.summary()

#%% Initialisation and training

# Initialisation
model.random_state_time_course_initialization(training_data, n_init=3, n_epochs=1)

# Full training
model.fit(training_data)
model.save(model_dir)

#%% Get inferred parameters

# State probabilities
alp = model.get_alpha(training_data)
pickle.dump(alp, open(inf_params_dir + "/alp.pkl", "wb"))

# State covariances
covs = model.get_covariances()
np.save(inf_params_dir + "/covs.npy", covs)
