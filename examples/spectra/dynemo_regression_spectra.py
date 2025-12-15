"""DyNeMo regression spectra."""

import os
import pickle
import numpy as np
from osl_dynamics.data import Data
from osl_dynamics.inference import modes
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
get_inf_params("tde_dynemo_notts_mrc_meguk_giles_5_subjects", rename="results/inf_params")

data = Data("source_data", n_jobs=5)
data = data.trim_time_series(n_embeddings=15, sequence_length=200)

alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

covs = np.load("results/inf_params/covs.npy")
alpha = modes.reweight_alphas(alpha, covs)

f, psd, coh, w = spectral.regression_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
    return_weights=True,
    n_jobs=5,
)
print("f.shape =", f.shape)
print("psd.shape =", psd.shape)
print("coh.shape =", coh.shape)
print("w.shape =", w.shape)
