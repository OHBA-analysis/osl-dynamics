"""Example script for making nice network plots you could put in a paper.

"""

import os
import numpy as np

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

#%% Get data

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

#%% Plot mean activity/power maps

# Load the inferred state means
means = np.load("notts_ae_hmm/means.npy")

# Make colour bar labels bigger
plotting.set_style({"xtick.labelsize": 16})

# Mean activity maps (this could be replace with power maps)
#
# For arguments that can be passed through plot_kwargs to change the plot see:
# https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_img_on_surf.html
power.save(
    means,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
    plot_kwargs={
        "views": ["lateral"],
        "symmetric_cbar": True,
    },
    filename="maps_.png",
)

#%% Plot graphical connectivity networks

# Load inferred AECs (which are learnt via the state covariance in an AE-HMM model)
covs = np.load("notts_ae_hmm/covs.npy")

# Do threshold (this will be different depending on how you calculated your conenctivity networks)
covs = connectivity.threshold(covs, percentile=98)

# Plot
#
# For arguments that can be passed through plot_kwargs to change the plot see:
# https://nilearn.github.io/stable/modules/generated/nilearn.plotting.view_connectome.html
#
# Or if glassbrain=False:
# https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_connectome.html
connectivity.save(
    covs,
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    plot_kwargs={},
    filename="conn_.html",
    glassbrain=True,
)