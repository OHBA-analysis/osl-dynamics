"""Example script for plotting functional connectivity (correlation) maps.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics.analysis import static, connectivity
from osl_dynamics.data import Data

# Make a directory to save plots to
os.makedirs("figures", exist_ok=True)

# Load data
data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 6)
    ]
)
ts = data.time_series()

# Calculate functional connectivity (Pearson correlation)
conn_map = static.functional_connectivity(ts)
conn_map = abs(conn_map)

# Plot subject-specific functional connectivities
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/fc_subj_.png",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    threshold=0.95,
)

# Plot group mean
conn_map = np.mean(conn_map, axis=0)
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/fc_group_.png",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    threshold=0.95,
)
