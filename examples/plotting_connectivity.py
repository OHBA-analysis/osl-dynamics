"""Example code for plotting connectivity.

"""

print("Setting up")
from vrad.analysis import connectivity, power, states
from vrad.data import OSL_HMM, io

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-55_K-12/hmm.mat"
)

n_embeddings = 15
pca_components = io.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo/"
    "natcomms18/prepared_data/pca_components.mat"
)

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Covariance of raw channels
raw_cov = states.raw_covariances(
    state_covariances=hmm.covariances,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
    zero_lag=False,
)

# Plot power maps
power.save(
    power_map=raw_cov,
    filename="power_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

# Plot connectivity
connectivity.save(
    connectivity_map=abs(raw_cov),
    threshold=0.98,
    filename="conn_.png",
    parcellation_file=parcellation_file,
)
