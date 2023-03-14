"""Post-hoc calculation of state spectra using a multitaper.

"""

import os
import pickle
import numpy as np

from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

# Directories
inf_params_dir = "results/inf_params"
spectra_dir = "results/spectra"

os.makedirs(spectra_dir, exist_ok=True)

#%% Load data and inferred state probabilities

# Load source reconstructed data
data = Data([f"data/subject{i}.npy" for i in range(1, 68)])

# Trim time point we lost during time-delay embedding and separating
# the data into sequences
data = data.trim_time_series(n_embeddings=15, sequence_length=2000)

# State probabilities
alpha = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

# Sanity check: the first axis should have the same numebr of time points
# for x, a in zip(data, alpha):
#    print(x.shape, a.shape)

#%% Calculate multitaper

f, psd, coh, w = spectral.multitaper_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,  # weighting for each subject when we average the spectra
    n_jobs=4,  # parallelisation
)

np.save(spectra_dir + "/f.npy", f)
np.save(spectra_dir + "/psd.npy", psd)
np.save(spectra_dir + "/coh.npy", coh)
np.save(spectra_dir + "/w.npy", w)

#%% Calculate non-negative matrix factorisation (NNMF)

# We fit 2 'wideband' components
wb_comp = spectral.decompose_spectra(coh, n_components=2)

np.save(spectra_dir + "/wb_comp.npy", wb_comp)
