"""Example code for plotting connectivity maps based on a correlation matrix.

In this example we use the inferred state/mode covariance matrices to
calculate the state/mode correlation matrix.

In this script we use an HMM fit, but this can be easily substituted for
a DyNeMo fit.
"""

import numpy as np

from osl_dynamics.array_ops import cov2corr
from osl_dynamics.analysis import connectivity
from osl_dynamics.data import OSL_HMM

# Load an HMM trained on amplitude envelope data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-8_zeroMean-1/hmm.mat"
)

# Get the inferred state covariances
covariances = hmm.covariances

# Convert the covariance matrix to a correlation matrix
correlations = cov2corr(covariances)

# Files used to source reconstruct the data the HMM was trained on
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

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
    connectivity_map=pos_conn_map,
    filename="pos_corr_.png",
    parcellation_file=parcellation_file,
)
connectivity.save(
    connectivity_map=neg_conn_map,
    filename="neg_corr_.png",
    parcellation_file=parcellation_file,
)
