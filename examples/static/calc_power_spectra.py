"""Example script for calculating static power spectra.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import static
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting

# Load data
data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 6)
    ]
)
ts = data.time_series()

# Calculate static power spectra
f, p = static.welch_spectra(
    data=ts,
    window_length=500,
    sampling_frequency=250,
    standardize=True,
)

# Average over channels
p = np.mean(p, axis=1)

# Plot
plotting.plot_line(
    [f] * p.shape[0],
    p,
    labels=[f"Subject {i + 1}" for i in range(p.shape[0])],
    filename="psd.png",
)
