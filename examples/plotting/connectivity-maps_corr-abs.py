"""Example code for plotting connectivity maps based on a correlation matrix.

In this example we use the inferred state/mode covariance matrices to
calculate the state/mode correlation matrices.

In this script we use an HMM fit, but this can be easily substituted for
a DyNeMo fit.
"""

print("Setting up")
import numpy as np
from osl_dynamics.array_ops import cov2corr
from osl_dynamics.analysis import connectivity, modes
from osl_dynamics.data import OSL_HMM

# Load an HMM trained on amplitude envelope data
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-8_zeroMean-1/hmm.mat"
)

# Get the inferred state covariances
covariances = hmm.covariances

# Convert the covariance matrix to a correlation matrix
correlations = cov2corr(covariances)

# We have many options for how to threshold the maps:
# - Plot the top X % of connections.
# - Plot the top X % of connections relative to the means across states/modes.
# - Use a GMM to calculate a threshold (X) for us.
#
# Here we will just plot the top 2% of the absolute values
conn_map = abs(correlations)
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.98,
    filename="corr_.png",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
)
