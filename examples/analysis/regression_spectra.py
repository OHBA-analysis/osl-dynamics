"""Example script for calculating mode spectra on real data
using the regression method.

"""

print("Setting up")
import pickle
import numpy as np
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

# Load the source reconstructed data
src_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 11)
    ]
)

# Get the source reconstructed data as a numpy array and remove time points
# loss due to to time delay embedding and separating into sequences
ts = src_data.trim_time_series(n_embeddings=15, sequence_length=200)

# Load inferred mixing coefficients from DyNeMo
alp = pickle.load(
    open(
        "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/alp.pkl",
        "rb",
    )
)

# Sanity check: make sure the length of alphas match the data
# for x, a in zip(ts, alp):
#     print(x.shape, a.shape)

# Calculate mode spectra
f, psd, coh, w = spectral.regression_spectra(
    data=ts,
    alpha=alp,
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
