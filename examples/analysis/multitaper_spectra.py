"""Example code for calculating a multitaper using a state time course
from an HMM fit.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import spectral
from osl_dynamics.data import OSL_HMM, Data

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

# Calculate subject-specific PSDs and coherences using multitaper method
f, psd, coh, w = spectral.multitaper_spectra(
    data=ts,
    alpha=alp,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,
)

# Group average
psd = np.average(psd, axis=0, weights=w)
coh = np.average(coh, axis=0, weights=w)

# Save the spectra
np.save("f.psd", f)
np.save("psd.psd", psd)
np.save("coh.psd", coh)
