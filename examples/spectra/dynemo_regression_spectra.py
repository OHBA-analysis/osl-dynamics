"""DyNeMo regression spectra."""

import os
import pickle
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

def get_inf_params(name, rename):
    os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

get_data("notts_meguk_giles_5_subjects", rename="source_data")
get_inf_params("tde_dynemo_notts_meguk_giles_5_subjects", rename="results/inf_params")

data = Data("source_data", n_jobs=5)
data = data.trim_time_series(n_embeddings=15, sequence_length=200)

alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

spectra = spectral.regression_spectra(
    data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    n_jobs=5,
)
print(spectra)
print("frequencies.shape =", spectra.frequencies.shape)
print("static_psd.shape =", spectra.static_psd.shape)
print("psd_coefs.shape =", spectra.psd_coefs.shape)
print("coherence_coefs.shape =", spectra.coherence_coefs.shape)
print("weights.shape =", spectra.weights.shape)
spectra.save("regression.npz")
