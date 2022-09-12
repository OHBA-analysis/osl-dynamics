"""Example code for plotting connectivity based on a correlation matrix.

In this example we use the inferred state/mode covariance matrices from
training on time-embedded/PCA data to calculate the state/mode correlation
matrices.

In this script we use an HMM fit, but this can be easily substituted for
a DyNeMo fit.
"""

print("Setting up")
import numpy as np
from osl_dynamics.array_ops import cov2corr
from osl_dynamics.analysis import connectivity, modes
from osl_dynamics.data import OSL_HMM, rw

# Load an HMM trained on time-delay embedded data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)

# Get the inferred covariances
# These covariances describe the training data in the time-delay embedded/PCA space
# We must convert this matrix into the source space by reversing the PCA and
# time-delay embedding
covs = hmm.covariances

# This is the number of embeddings and PCA components used to prepare the training data
n_embeddings = 15
pca_components = rw.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/prepared_data/pca_components.mat"
)

# Files used to create the source reconstructed data
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Convert from the time-delay embedded/PCA space to the original source space
covs = modes.raw_covariances(
    covs,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
    zero_lag=False,  # Should we just consider the zero-lag covariance?
)

# Convert from covariance to correlation
corrs = cov2corr(covs)

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will just plot the top 2% of the absolute values
connectivity.save(
    connectivity_map=abs(corrs),
    filename="corr_.png",
    threshold=0.98,
    parcellation_file=parcellation_file,
)
