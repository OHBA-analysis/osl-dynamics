"""Calculate static spectra.

Note: the static.welch_spectra function can be substituted below
to use Welch's method for calculating spectra.
"""

import os
from osl_dynamics.data import Data
from osl_dynamics.analysis import static

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

get_data("notts_meguk_giles_5_subjects", rename="source_data")

data = Data("source_data", n_jobs=5)
data = data.time_series()

# Static PSDs, coherences and cross spectra
spectra = static.multitaper_spectra(
    data,
    sampling_frequency=250,
    frequency_range=[1, 45],
    n_jobs=5,
)
print()
print(spectra)
print("frequencies.shape =", spectra.frequencies.shape)
print("psd.shape =", spectra.psd.shape)
print("coherence.shape =", spectra.coherence.shape)
print("cross_spectra.shape =", spectra.cross_spectra.shape)
print("session_weights.shape =", spectra.session_weights.shape)
print()

# Static PSDs only
spectra = static.multitaper_spectra(
    data,
    sampling_frequency=250,
    frequency_range=[1, 45],
    cross_spectra=False,
    n_jobs=5,
)
print(spectra)
print("psd.shape =", spectra.psd.shape)
