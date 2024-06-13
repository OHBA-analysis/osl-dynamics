"""Example script for plotting the activity maps from the state/mode means.

In this example we plot the state means from an HMM fit, but this can be
easily substituted with mode means from DyNeMo.
"""

import os
import numpy as np

from osl_dynamics.analysis import power

def get_data(name, output_dir):
    if os.path.exists(output_dir):
        print(f"{output_dir} already downloaded. Skipping..")
        return
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {output_dir}")
    os.remove(f"{name}.zip")
    print(f"Data downloaded to: {output_dir}")

# We will download example data hosted on osf.io/by2tc.
get_data("ae_hmm_notts_mrc_meguk_glasser", output_dir="notts_ae_hmm")

# Load the inferred state means
means = np.load("notts_ae_hmm/means.npy")

# Mean activity maps
power.save(
    means,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    subtract_mean=True,
    filename="maps_.png",
)
