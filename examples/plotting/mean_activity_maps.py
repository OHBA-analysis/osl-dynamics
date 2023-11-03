"""Example script for plotting the activity maps from the state/mode means.

In this example we plot the state means from an HMM fit, but this can be
easily substituted with mode means from DyNeMo.
"""

import os
import numpy as np

from osl_dynamics.analysis import power

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

# Load the inferred state means
means = np.load("notts_ae_hmm/means.npy")

# Mean activity maps
power.save(
    means,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
    filename="maps_.png",
)
