"""Example code for calculating partial covariances using an HMM fit.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import modes
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

# Calculate partial covariances and save
pcovs = modes.partial_covariances(ts, alp)

np.save("pcovs.npy", pcovs)
