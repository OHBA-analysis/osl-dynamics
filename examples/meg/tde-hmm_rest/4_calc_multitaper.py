"""Post-hoc calculation of state-spectra using a multitaper.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 4_calc_multitaper.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import pickle
import numpy as np

from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

#%% Directories

# Directories
inf_params_dir = f"results/{n_states}_states/run{run:02d}/inf_params"
spectra_dir = f"results/{n_states}_states/run{run:02d}/spectra"

os.makedirs(spectra_dir, exist_ok=True)

#%% Load data and inferred state probabilities

# Load source reconstructed data
data = Data(
    "training_data/networks",
    store_dir=f"tmp_{n_states}_{run}",
    n_jobs=8,
)

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

# Sanity check: the first axis should have the same numebr of time points
#for x, a in zip(data, alpha):
#   print(x.shape, a.shape)

#%% Calculate multitaper

f, psd, coh, w = spectral.multitaper_spectra(
    data=data_,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,  # weighting for each subject when we average the spectra
    n_jobs=8,  # parallelisation, if you get a RuntimeError set to 1
)

np.save(f"{spectra_dir}/f.npy", f)
np.save(f"{spectra_dir}/psd.npy", psd)
np.save(f"{spectra_dir}/coh.npy", coh)
np.save(f"{spectra_dir}/w.npy", w)

#%% Calculate non-negative matrix factorisation (NNMF)

# We fit 2 'wideband' components
wb_comp = spectral.decompose_spectra(coh, n_components=2)

np.save(f"{spectra_dir}/nnmf_2.npy", wb_comp)
