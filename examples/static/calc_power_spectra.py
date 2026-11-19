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
spectra = static.welch_spectra(
    ts,
    sampling_frequency=250,
    window_length=500,
    cross_spectra=False,
)

# Average over channels
p = np.mean(spectra.psd, axis=1)

# Plot
plotting.plot_line(
    [spectra.frequencies] * p.shape[0],
    p,
    labels=[f"Subject {i + 1}" for i in range(p.shape[0])],
    filename="psd.png",
)
