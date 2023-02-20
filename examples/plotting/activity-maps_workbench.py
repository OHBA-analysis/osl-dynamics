"""Example code for saving maps as nii files and plotting with workbench.

"""

from osl_dynamics.analysis import power, workbench
from osl_dynamics.data import OSL_HMM

# Load an HMM trained on amplitude envelope data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-6_zeroMean-0/hmm.mat"
)

# Mean activity maps (state means)
means = hmm.means

# Source reconstruction files used to create the data the HMM was trained on
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Save nii file
power.save(
    power_map=means,
    filename="maps.nii.gz",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

# Setup workbench by specifying the path it is located
workbench.setup("/well/woolrich/projects/software/workbench/bin_rh_linux64")

# Use workbench to save the maps in the nii file as images
# "tmp" is a directory used to store temporary files created by workbench
# this directory can be safely deleted after the maps have been saved
workbench.render("maps.nii.gz", "tmp", gui=False, image_name="maps_.png")
