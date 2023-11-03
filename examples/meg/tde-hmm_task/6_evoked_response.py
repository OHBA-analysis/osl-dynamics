"""Group-average evoked response analysis.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 6_evoked_response.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import mne
import pickle
import numpy as np
from glob import glob

from osl_dynamics.analysis import statistics
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

#%% Setup directories

results_dir = f"results/{n_states}_states/run{run:02d}"
inf_params_dir = f"{results_dir}/inf_params"
evoked_response_dir = f"{results_dir}/evoked_response"

os.makedirs(evoked_response_dir, exist_ok=True)

#%% Load data

# State time courses
alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcellated data files
parc_files = sorted(glob("data/src/*/sflip_parc-raw.fif"))

#%% Epoch

# Task info
event_id = {
    "famous/first": 5,
    "famous/immediate": 6,
    "famous/last": 7,
    "unfamiliar/first": 13,
    "unfamiliar/immediate": 14,
    "unfamiliar/last": 15,
    "scrambled/first": 17,
    "scrambled/immediate": 18,
    "scrambled/last": 19,
}

# Window to epoch around
tmin = -2
tmax = 5

# Epoch the state time course
epochs = []
for s, p in zip(stc, parc_files):
    raw = modes.convert_to_mne_raw(s, p, n_embeddings=15)
    events = mne.find_events(raw, min_duration=0.005, verbose=False)
    e = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
    )
    ae = np.mean(e.get_data(picks="misc"), axis=0).T  # average over trials
    epochs.append(ae)
epochs = np.array(epochs)  # (subjects, time, states)

# Baseline correct using the average value from t=tmin to t=0
epochs -= np.mean(
    epochs[:, : int(abs(tmin) * raw.info["sfreq"])],
    axis=1,
    keepdims=True,
)

#%% Stats testing

# Calculate p-values
pvalues = statistics.evoked_response_max_stat_perm(
    epochs,
    n_perm=1000,
    metric="copes",
    n_jobs=8,
)

# Plot
plotting.plot_evoked_response(
    e.times,
    np.mean(epochs, axis=0),  # average over subjects
    pvalues,
    significance_level=0.05,
    labels=[f"State {i + 1}" for i in range(epochs.shape[-1])],
    x_label="Time (s)",
    y_label="State Probability",
    filename=f"{evoked_response_dir}/epoched_stc.png",
)
