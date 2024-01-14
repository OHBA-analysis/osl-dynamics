"""Calculate static (time-averaged) features from neuroimaging data.

Note, the data in "data/src" isn't provided. You can use the get_data.py
script in the parent directory to download the output of this script.
"""

import os
import numpy as np
import pandas as pd
from glob import glob

from osl_dynamics.data import Data
from osl_dynamics.analysis import static, power

#%% Load data

files = sorted(glob("data/src/sub-*/sflip_parc-raw.fif"))
data = Data(
    files,
    picks="misc",
    reject_by_annotation="omit",
    sampling_frequency=250,
    load_memmaps=False,
    n_jobs=8,
)

#%% Calculate power in canonical frequency bands

bands = [[1, 4], [4, 8], [8, 13], [13, 30]]

x = data.time_series()
f, psd = static.welch_spectra(
    data=x,
    window_length=500,
    sampling_frequency=250,
    n_jobs=8,
)
p = [power.variance_from_spectra(f, psd, frequency_range=b) for b in bands]

#%% Calculate AEC in canonical frequency bands

aec = []
for b in bands:
    methods = {
        "filter": {"low_freq": b[0], "high_freq": b[1], "use_raw": True},
        "amplitude_envelope": {},
        "standardize": {},
    }
    data.prepare(methods)
    x = data.time_series()
    aec.append(static.functional_connectivity(x, conn_type="corr"))

#%% Calculate TDE covariances

methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 100, "use_raw" : True},
    "standardize": {},
}
data.prepare(methods)
x = data.time_series()
tde_cov = static.functional_connectivity(x, conn_type="cov")

#%% Save

# Reshape: (frequency_bands, subjects, ...) -> (subjects, frequency_bands, ...)
power = np.swapaxes(power, 0, 1)
aec = np.swapaxes(aec, 0, 1)

os.makedirs("data/static_features", exist_ok=True)
np.save("data/static_features/power.npy", p)
np.save("data/static_features/aec.npy", aec)
np.save("data/static_features/tde_cov.npy", tde_cov)

#%% Get ages for each subject

# Subject IDs
subjects = [file.split("/")[-2] for file in files]

# Get demographics
participants = pd.read_csv("camcan/participants.tsv", sep="\t")

# Get ages
ages = []
for id in subjects:
    age = participants.loc[participants["participant_id"] == id]["age"].values[0]
    ages.append(age)

# Save
np.save("data/age.npy", ages)
