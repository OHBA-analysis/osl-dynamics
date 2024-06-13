"""Example code for plotting connectivity maps based on a correlation matrix.

In this example we use the inferred state/mode covariance matrices to
calculate the state/mode correlation matrices.

In this script we use an HMM fit, but this can be easily substituted for
a DyNeMo fit.
"""

import os
import numpy as np

from osl_dynamics.array_ops import cov2corr
from osl_dynamics.analysis import connectivity

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

# Load the inferred state covariances
covariances = np.load("notts_ae_hmm/covs.npy")

# Convert the covariance matrix to a correlation matrix
correlations = cov2corr(covariances)

# Take the absolute value
conn_map = abs(correlations)

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will just plot the top 2% of the absolute values
connectivity.save(
    conn_map,
    threshold=0.98,
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    filename="corr_.png",
)
