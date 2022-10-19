"""Example script for calculating static power/coherence spectra.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import static
from osl_dynamics.data import Data

# Load data
data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 6)
    ]
)
ts = data.time_series()

# Calculate static power spectra
f, p, c = static.power_spectra(
    data=ts,
    window_length=500,
    sampling_frequency=250,
    standardize=True,
    calc_coh=True,
)

# Save to plot power/coherence maps: see static/plot_maps.py
np.save("f.npy", f)
np.save("p.npy", p)
np.save("c.npy", c)
