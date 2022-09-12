"""Example script for calculating state/mode spectra on real data
using the regression method.

In this example we use an HMM fit, but this can be substituted with
a DyNeMo fit.
"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data, OSL_HMM

# Load the source reconstructed data
src_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 11)
    ]
)

# Get the source reconstructed data as a numpy array and remove time points
# loss due to to time delay embedding
ts = src_data.trim_time_series(n_embeddings=15)

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)

# Get the inferred state time course
alp = hmm.state_time_course()

# Sanity check: make sure the length of alphas match the data
# for i in range(len(ts)):
#     print(ts[i].shape, alp.shape[i])

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
)

# Group average
psd = np.average(psd, axis=0, weights=w)
coh = np.average(coh, axis=0, weights=w)

# Save
np.save("f.npy", f)
np.save("psd.npy", psd)
np.save("coh.npy", coh)
