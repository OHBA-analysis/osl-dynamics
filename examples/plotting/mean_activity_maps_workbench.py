"""Example code for saving maps as nii files and plotting with workbench.

"""

import os
import numpy as np

from osl_dynamics.analysis import power, workbench

# We will download example data hosted on osf.io/by2tc.
# Note, osfclient must be installed. This can be installed with pip:
#
#     pip install osfclient

def get_data(name, output_dir):
    if os.path.exists(output_dir):
        print(f"{output_dir} already downloaded. Skipping..")
        return
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {output_dir}")
    os.remove(f"{name}.zip")
    print(f"Data downloaded to: {output_dir}")

get_data("ae-hmm_notts_rest_55_subj", output_dir="notts_ae_hmm")

# Load mean activity maps (state means)
means = np.load("notts_ae_hmm/means.npy")

# Save nii file
power.save(
    means,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
    filename="maps.nii.gz",
)

# Setup workbench by specifying the path it is located
workbench.setup("/well/woolrich/projects/software/workbench/bin_rh_linux64")

# Use workbench to save the maps in the nii file as images
# "tmp" is a directory used to store temporary files created by workbench
# this directory can be safely deleted after the maps have been saved
workbench.render("maps.nii.gz", "tmp", gui=False, image_name="maps_.png")
