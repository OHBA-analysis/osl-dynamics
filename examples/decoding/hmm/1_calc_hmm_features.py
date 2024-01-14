"""Save HMM extract features.

The data needed to run this script has not been provide. You the get_data.py
script in the parent directory to download the output of this script.
"""

import pickle
import numpy as np
import pandas as pd
from glob import glob

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.inference import modes

#%% Calculate HMM features summarising dynamics

# Load inferred state probabilities
alp = pickle.load(open("data/inf_params/alp.pkl", "rb"))

# Hard classify
stc = modes.argmax_time_courses(alp)

# Summary statistics
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250)
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)

# Save
np.save("data/hmm_features/fo.npy", fo)
np.save("data/hmm_features/lt.npy", lt)
np.save("data/hmm_features/intv.npy", intv)
np.save("data/hmm_features/sr.npy", sr)

#%% Calculate multitaper power maps and coherence networks

# Load multitaper spectra
f = np.load("data/spectra/f.npy")
psd = np.load("data/spectra/psd.npy")
coh = np.load("data/spectra/coh.npy")

# Calculate power maps
power_maps = power.variance_from_spectra(f, psd)

# Calculate coherence networks (only keeping the upper triangle)
coherence_networks = connectivity.mean_coherence_from_spectra(f, coh)
m, n = np.triu_indices(coherence_networks.shape[-1], k=1)
coherence_networks = coherence_networks[..., m, n]

# Save
np.save("data/hmm_features/power_maps.npy", power_maps)
np.save("data/hmm_features/coherence_networks.npy", coherence_networks)

#%% Get ages for each subject

# Subject IDs
files = sorted(glob("data/src/sub-*/sflip_parc-raw.fif"))
subjects = [file.split("/")[-2] for file in files]

# Get demographics
participants = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")

# Get ages
ages = []
for id in subjects:
    age = participants.loc[participants["participant_id"] == id]["age"].values[0]
    ages.append(age)

# Save
np.save("data/age.npy", ages)
