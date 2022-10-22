"""Example code for plotting power maps from the state/mode covariances inferred
on time-delay embedded training data.

In this example we use the inferred matrix directly.
"""

print("Setting up")
from osl_dynamics.analysis import power, modes
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
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Convert from the time-delay embedded/PCA space to the original source space
power_map = modes.raw_covariances(
    covs,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
)

# Save the power maps
power.save(
    power_map=power_map,
    filename="maps_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)
