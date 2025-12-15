"""Calculate HMM state spectra using a multitaper."""

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

get_data("notts_mrc_meguk_giles_5_subjects", rename="source_data")
get_inf_params("tde_hmm_notts_mrc_meguk_giles_5_subjects", rename="results/inf_params")

data = Data("source_data", n_jobs=5)
data = data.trim_time_series(n_embeddings=15, sequence_length=2000)

alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

f, psd, coh, w = spectral.multitaper_spectra(
    data=data,
    alpha=alpha,
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
