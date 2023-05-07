"""Group average evoked response analysis.

"""

import mne
import pickle
import numpy as np
from glob import glob

from osl_dynamics.analysis import statistics
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

# Directories
inf_params_dir = "results/inf_params"
plots_dir = "results/plots"

#%% Load data

# State time courses
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcellated data files
parc_files = sorted(glob("data/src/*/sflip_parc-raw.fif"))

#%% Task info

keys = ["lowPower_Grip", "highPower_Grip"]
values = [1, 2]
event_id = dict(zip(keys, values))

#%% Epoch

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

# Do statistical significance testing
pvalues = statistics.evoked_response_max_stat_perm(
    epochs, n_perm=1000, metric="copes", n_jobs=4
)

# Plot
plotting.plot_evoked_response(
    e.times,
    np.mean(epochs, axis=0),  # average over subjects
    pvalues,
    significance_level=0.05,
    labels=[f"State {i + 1}" for i in range(epochs.shape[-1])],
    x_label="Time (s)",
    y_label="State Activation",
    filename=plots_dir + "/epoched_stc.png",
)
