"""Example code for calculating a multitaper using a state time course
from an HMM fit.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data, HMM_MAR

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

# Load an HMM fit from the MATLAB toolbox
hmm = HMM_MAR(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)

# Get the inferred state time course
alp = hmm.state_time_course()

# Sanity check: make sure the length of alphas match the data
# for x, a in zip(ts, alp):
#     print(x.shape, a.shape)

# Calculate subject-specific PSDs and coherences using multitaper method
f, psd, coh, w = spectral.multitaper_spectra(
    data=ts,
    alpha=alp,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,
    n_jobs=5,
)

# Save the spectra
np.save("f.npy", f)
np.save("psd.npy", psd)
np.save("coh.npy", coh)
np.save("w..npy", w)
