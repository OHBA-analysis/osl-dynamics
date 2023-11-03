"""Example script for plotting Amplitude Envelope Correlation (AEC) functional connectivity maps.

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
    ],
    sampling_frequency=250,
)

# Use prepare data to compute AE
methods = {
    "filter": {"low_freq": 8, "high_freq": 12},
    "amplitude_envelope": {},
    "standardize": {},
}
data.prepare(methods)
ts = data.time_series()

# Calculate functional connectivity using AEC
conn_map = static.functional_connectivity(ts)

# Plot group mean AEC
conn_map = np.mean(conn_map, axis=0)
connectivity.save(
    connectivity_map=conn_map,
    filename="figures/aec_group_.png",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    threshold=0.98,
)
