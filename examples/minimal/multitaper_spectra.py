"""Calculate a multitaper spectrum and non-negative matrix factorization (NNMF).

This is the recommended approach the HMM.
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

# Get the inferred state probabilities from the HMM
#
# We saved these using the get_inf_params.py script.
alpha = pickle.load(open("model/alp.pkl", "rb"))

# Sanity check:
# - Make sure the length of alphas match the data
# - If they don't match, something's gone wrong
for x, a in zip(data, alpha):
    print(x.shape, a.shape)

# Calculate subject-specific PSDs and coherences using multitaper method
f, psd, coh, w = spectral.multitaper_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,
    n_jobs=5,
)

# Calculate NNMF
wb_comp = spectral.decompose_spectra(coh, n_components=2)

# Save the spectra
np.save("f.npy", f)
np.save("psd.npy", psd)
np.save("coh.npy", coh)
np.save("w..npy", w)
np.save("wb_comp.npy", wb_comp)
