"""Group average evoked response analysis.

"""

import pickle
import numpy as np

from osl_dynamics.data import task
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

# Directories
inf_params_dir = "results/inf_params"
plots_dir = "results/plots"

#%% Load state time course

# Load state probabilities
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

# Take the most probable state at each time point (Viterbi path)
stc = modes.argmax_time_courses(alp)

#%% Epoch the state time courses

pre = 2 * 250  # time before the event to epoch in samples
post = 5 * 250  # time after the event to epoch in samples

epochs = []
for s in stc:
    # First event is at index 500 - 7 = 493
    # This is because we lost the first 7 time points due to time embedding
    # Every other event is 1751 time points after (from the preprocessing)
    event_indices = np.arange(2 * 250 - 7, s.shape[0], 7 * 250 + 1)

    # Average over epochs for this subject
    subject_average_epoch = task.epoch_mean(s, event_indices, pre=pre, post=post)

    epochs.append(subject_average_epoch)

# Average over subjects
epoched_stc = np.mean(epochs, axis=0)

# Baseline correct
epoched_stc -= np.mean(epoched_stc[:pre], axis=0, keepdims=True)

# Plot
t = np.arange(-pre, post) / 250
n_states = epoched_stc.shape[1]
plotting.plot_line(
    [t] * n_states,
    epoched_stc.T,
    labels=[f"State {i}" for i in range(1, n_states + 1)],
    legend_loc=2,
    x_label="Time (s)",
    y_label="Average State Activation",
    filename=plots_dir + "/epoched_stc.png",
)
