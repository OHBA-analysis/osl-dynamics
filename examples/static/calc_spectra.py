"""Example script for calculating static power/coherence spectra.

"""

print("Setting up")
import os
from osl_dynamics.analysis import static
from osl_dynamics.data import Data

# Make a directory to hold output files
os.makedirs("spectra", exist_ok=True)

# Load data
data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 6)
    ]
)
ts = data.time_series()

# Calculate static power/coherence spectra
spectra = static.welch_spectra(
    ts,
    sampling_frequency=250,
    window_length=500,
)

# Save to plot power/coherence maps: see static/plot_*_maps.py
spectra.save("spectra/static.npz")
