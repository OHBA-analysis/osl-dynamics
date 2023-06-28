"""Example code for plotting power maps from the state/mode covariances inferred
on time-delay embedded training data.

"""

import os
import numpy as np

from osl_dynamics.analysis import power, modes
from osl_dynamics.data import rw

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

get_data("tde-hmm_notts_rest_55_subj", output_dir="notts_tde_hmm")

# Get the inferred covariances
#
# These covariances describe the training data in the time-delay embedded/PCA space
# We must convert this matrix into the source space by reversing the PCA and
# time-delay embedding
covs = np.load("notts_tde_hmm/covs.npy")

# This is the number of embeddings and PCA components used to prepare the training data
n_embeddings = 15
pca_components = rw.loadmat("notts_tde_hmm/pca_components.mat")

# Convert from the time-delay embedded/PCA space to the original source space
power_map = modes.raw_covariances(
    covs,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
)

# Save the power maps
power.save(
    power_map,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
    filename="maps_.png",
)
