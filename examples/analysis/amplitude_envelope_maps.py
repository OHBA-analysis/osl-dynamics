"""Example script for plotting the mean activity and correlation maps inferred by
fitting an HMM to amplitude envelope data.

"""

import os
from osl_dynamics.array_ops import cov2corr
from osl_dynamics.data import OSL_HMM
from osl_dynamics.analysis import connectivity, power

# Create a directory to save plots
os.makedirs("figures", exist_ok=True)

# Load HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/summer21/results/Subj1-55_K-8_zeroMean-0/hmm.mat"
)
cov = hmm.covariances

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Mean activity maps
power.save(
    cov,
    filename="figures/mean_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

# Correlation maps
conn_map = cov2corr(cov)
conn_map = connectivity.threshold(conn_map, percentile=90, absolute_value=True)
pos_conn_map, neg_conn_map = connectivity.separate_edges(conn_map)
connectivity.save(
    pos_conn_map,
    filename="figures/pos_corr_.png",
    parcellation_file=parcellation_file,
)
connectivity.save(
    neg_conn_map,
    filename="figures/neg_corr_.png",
    parcellation_file=parcellation_file,
)
