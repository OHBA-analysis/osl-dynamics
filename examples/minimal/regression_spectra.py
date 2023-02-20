"""Example script for calculating mode spectra using the GLM-spectrum approach.

This is the recommended approach with a DyNeMo model.
"""

import pickle
import numpy as np

from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

# Load training data (in the same way as the prepare_data.py script)
#
# Note, this has to be the unprepared data (i.e. before time-delay embedding/PCA)
files = [f"subject{i}.npy" for i in range(5)]
data = Data(files)

# We need to remove data points lost due to time-delay embedding and separating
# into sequences
data = data.trim_time_series(n_embeddings=15, sequence_length=200)

# Get the inferred mixing coefficients from DyNeMo
#
# We saved these using the get_inf_params.py script.
alpha = pickle.load(open("model/alp.pkl", "rb"))

# Sanity check:
# - Make sure the length of alphas match the data
# - If they don't match, something's gone wrong
for x, a in zip(data, alpha):
    print(x.shape, a.shape)

# Calculate mode spectra
f, psd, coh, w = spectral.regression_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    window_length=1000,
    frequency_range=[0, 45],
    step_size=20,
    n_sub_windows=8,
    return_weights=True,
    return_coef_int=True,
    n_jobs=5,
)

# Save
np.save("f.npy", f)
np.save("psd.npy", psd)
np.save("coh.npy", coh)
np.save("w.npy", w)
