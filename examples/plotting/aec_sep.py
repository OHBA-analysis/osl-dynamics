"""Example code for plotting connectivity maps based on a correlation matrix.

In this example we use the inferred state/mode covariance matrices to
calculate the state/mode correlation matrix.

In this script we use an HMM fit, but this can be easily substituted for
a DyNeMo fit.
"""

import os
import numpy as np

from osl_dynamics.array_ops import cov2corr
from osl_dynamics.analysis import connectivity

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

# Load the inferred state covariances
covariances = np.load("notts_ae_hmm/covs.npy")

# Convert the covariance matrix to a correlation matrix
correlations = cov2corr(covariances)

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will plot the top 5% of connections (irrespective of sign and relative
# to the mean) and we will separate maps for the postive and negative edges
conn_map = connectivity.threshold(
    correlations, percentile=95, absolute_value=True, subtract_mean=True
)
pos_conn_map, neg_conn_map = connectivity.separate_edges(conn_map)
connectivity.save(
    pos_conn_map,
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    filename="pos_corr_.png",
)
connectivity.save(
    neg_conn_map,
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    filename="neg_corr_.png",
)
