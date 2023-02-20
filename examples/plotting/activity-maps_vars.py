"""Example script for plotting the activity maps from the state/mode covariances.

In this example our measure of activity is the variance (i.e. power) which we
extract from the diagonal of the covariance matrix directly. This approach is
typically used when we train with zero mean.
"""

from osl_dynamics.analysis import power
from osl_dynamics.data import OSL_HMM

# Load an HMM trained on amplitude envelope data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-8_zeroMean-1/hmm.mat"
)

# Get the inferred state covariances
covs = hmm.covariances

# Files used during source reconstruction
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Plot power maps (this function will automatically extract the diagonal)
power.save(
    covs,
    filename="maps_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)
