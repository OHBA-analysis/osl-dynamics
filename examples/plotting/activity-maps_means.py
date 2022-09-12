"""Example script for plotting the activity maps from the state/mode means.

In this example we plot the state means from an HMM fit, but this can be
easily substituted with mode means from DyNeMo.
"""

print("Setting up")
from osl_dynamics.analysis import power
from osl_dynamics.data import OSL_HMM

# Load an HMM trained on amplitude envelope data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-8_zeroMean-0/hmm.mat"
)

# Get the inferred state means
means = hmm.means

# Files used to source reconstruct the data the HMM was trained on
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Mean activity maps
power.save(
    means,
    filename="maps_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)
