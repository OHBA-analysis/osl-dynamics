"""Post-hoc calculation of mode-spectra using a GLM regression.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of modes and run id, e.g. python 4_calc_regression_spectra.py 6 1")
    exit()
n_modes = int(argv[1])
run = int(argv[2])

#%% Import packages

import os
import pickle
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

#%% Directories

inf_params_dir = f"results/{n_modes}_modes/run{run:02d}/inf_params"
spectra_dir = f"results/{n_modes}_modes/run{run:02d}/spectra"

os.makedirs(spectra_dir, exist_ok=True)

#%% Load data and inferred state probabilities

# Load source reconstructed data
data = Data("training_data", n_jobs=8)

# Trim time point we lost during time-delay embedding and separating
# the data into sequences
#
# Note:
# - n_embeddings must be the same as that used to prepare the training
#   data.
# - sequence_length must be the same as used in the Config to create
#   the model.
data_ = data.trim_time_series(n_embeddings=15, sequence_length=200)

# State probabilities
alpha = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))

# Sanity check: the first axis should have the same number of time points
#for x, a in zip(data_, alpha):
#   print(x.shape, a.shape)

#%% Calculate spectra

spectra = spectral.regression_spectra(
    data_,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    n_jobs=8,
)
spectra.save(f"{spectra_dir}/regression.npz")
