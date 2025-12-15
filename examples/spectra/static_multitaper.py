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

get_data("notts_mrc_meguk_giles_5_subjects", rename="source_data")

data = Data("source_data", n_jobs=5)
data = data.time_series()

# Static PSDs only
f, psd, w = static.multitaper_spectra(
    data,
    sampling_frequency=250,
    frequency_range=[1, 45],
    return_weights=True,
    n_jobs=5,
)
print()
print("f.shape =", f.shape)
print("psd.shape =", psd.shape)
print("w.shape =", w.shape)
print()

# Static PSDs and coherences
f, psd, coh, w = static.multitaper_spectra(
    data,
    sampling_frequency=250,
    frequency_range=[1, 45],
    return_weights=True,
    calc_coh=True,
    n_jobs=5,
)
print("f.shape =", f.shape)
print("psd.shape =", psd.shape)
print("coh.shape =", coh.shape)
print("w.shape =", w.shape)
print()

# Static cross PSDs
f, cpsd, w = static.multitaper_spectra(
    data,
    sampling_frequency=250,
    frequency_range=[1, 45],
    return_weights=True,
    calc_cpsd=True,
    n_jobs=5,
)
print("f.shape =", f.shape)
print("cpsd.shape =", cpsd.shape)
print("w.shape", w.shape)
