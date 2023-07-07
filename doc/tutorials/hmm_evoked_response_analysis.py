"""
HMM: Evoked Response Analysis
=============================

In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on task source reconstructed MEG data. This tutorial covers:

1. Download a Trained Model and Data
2. Epoching the Raw Data
3. Epoching the HMM State Time Course

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/agrvz>`_ for the expected output.
"""

#%%
# Download a Trained Model and Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download a trained model
# ************************
# First, let's download a model that's already been trained on a task dataset. See the `HMM Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_training_real_data.html>`_ for how to train an HMM.

import os

def get_trained_model(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Model downloaded to: {name}"

# Download the trained model (approximately 80 MB)
model_name = "hmm_notts_task_10_subj"
get_trained_model(model_name)

# List the contents of the downloaded directory
sub_dirs = os.listdir(model_name)
print(sub_dirs)
for sub_dir in sub_dirs:
    print(f"{sub_dir}:", os.listdir(f"{model_name}/{sub_dir}"))

#%%
# Download the training data
# **************************
# Let's also download the data this model was trained on. This data was recorded during a visuomotor task. Where the subject alternated between being presented with a visual stimulus and performing a motor task.

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 150 MB)
get_data("notts_task_10_subj")

# List the contents of the downloaded directory containing the dataset
os.listdir("notts_task_10_subj")

#%%
# We can see the dataset contains numpy files for 10 subjects and a pickle file called `events.pkl`. Let's load the subject data using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("notts_task_10_subj")
print(data)

#%%
# And let's see what's in the pickle file.

import pickle

events = pickle.load(open("notts_task_10_subj/events.pkl", "rb"))
print(type(events))
print(events.keys())

#%%
# We can see events is a python `dict` with a `visual` and `motor` key, we'll discuss this variable in the next section. In this tutorial, we'll just focus on the visual events.
#
# Epoching the Raw Data
# ^^^^^^^^^^^^^^^^^^^^^
# Let's see if we can observe the visual event in the raw data. The `events.pkl` object contains numpy arrays containing the indices when each event occurred. Let's extract the event timings.

# Unpack the visual task timings
event_indices = events["visual"]

#%%
# The `event_indices` variable is a list of numpy arrays, one for each subject. E.g. we can see the indices when the visual task occurs for subject 1 by printing `event_indices[0]`:

print(event_indices[0])

#%%
# Now we know when the events occur, let's epoch around each event. osl-dynamics has the `data.task.epoch <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/task/index.html#osl_dynamics.data.task.epoch>`_ function to do this for us. To use this function we need to pass:
#
# - The time series we want to epoch.
# - A 1D array containing the sample number (index) when each event occurs.
# - `pre`: the number of samples before the event we want to keep.
# - `post`: the number of events after the event we want to keep.

from osl_dynamics.data import task

# Get the source data time series
ts = data.time_series()

# Epoch
ts_epochs = []
for x, v in zip(ts, event_indices):
    ts_epochs.append(task.epoch(x, v, pre=250, post=500))

#%%
# `ts_epochs` is a list of 3D arrays, one for each subject. The shape of each 3D array is `(n_epochs, epoch_length, n_channels)`. We can print how many epochs each subject has by printing the shape.

for v in ts_epochs:
    print(v.shape)

#%%
# We see each subject has roughly 40 trials for the visual task.
#
# For our current analysis we'll do a group analysis so we don't need to worry about the subject, we can pretend all trials belong to a single subject. Let's concatenate across subjects.

import numpy as np

concat_ts_epochs = np.concatenate(ts_epochs)
print(concat_ts_epochs.shape)

#%%
# We see we have 391 epochs for the task.
#
# To check we have epoched the data correctly, it's useful to plot the time series averaged over epochs. Let's plot the average value across channels and epochs. Note, by default `task.epochs` will fill values we don't have the full epoch for with `nan`s. Consequently, we should use `np.nanmean` when we average over epochs.

from osl_dynamics.utils import plotting

# Average over epochs and channels
avg_ts_epoch = np.nanmean(concat_ts_epochs, axis=(0,2))

# Plot
fs = 250  # sampling frequency in Hz
t = (np.arange(avg_ts_epoch.shape[0]) - 250) / fs  # time axis
fig, ax = plotting.plot_line(
    [t],
    [avg_ts_epoch],
    x_label="Time (s)",
    y_label="Signal (a.u.)",
)
ax.axvline(color="r", linestyle="--")

#%%
# We can see there's a clear response to the task, which gives us confidence the task indices are correct. Note, the visual response is quite strong. Depending on your task you may not see such a clean response in the raw data.
#
# Another thing we could do is plot the average value of the signal during the response to the task. Let's first look at the window around the peak. Let's highlight the window in the plot.

# Get indices for the time window
t_start = 0.1
t_end = 0.2
window_start = np.squeeze(np.argwhere(t == t_start))
window_end = np.squeeze(np.argwhere(t == t_end))
print("indices:", window_start, window_end)

# Highlight the window in the plot
fig, ax = plotting.plot_line(
    [t],
    [avg_ts_epoch],
    x_label="Time (s)",
    y_label="Signal (a.u.)",
)
ax.axvline(color="r", linestyle="--")
ax.axvspan(t_start, t_end, color="g", alpha=0.2)

#%%
# Now, let's calculate the average signal at each channel in this window and plot as a heat map.

from osl_dynamics.utils import plotting

# Get the average signal (averaging over epochs and the window)
window_start = int(window_start)
window_end = int(window_end)
ts_response = np.nanmean(concat_ts_epochs[:, window_start:window_end], axis=(0, 1))

# Plot as a heat map (takes a few seconds to appear)
fig, ax = plotting.plot_brain_surface(
    ts_response,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We see there is a response is in the visual cortex but the response isn't very clean.
#
# Epoching the HMM State Time Course
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, let's see what effect the visual task had on the brain networks that have been inferred by the HMM. Note, when we trained the HMM, it was unsupervised, meaning it had no knowledge of the task occurring.
#
# Get the inferred state time course
# **********************************
# Let's first get the inferred state time course. The probability of the state at each time point was saved in `data/alpha.pkl` when we trained the HMM, so all we need to do load this and take the most probable state. Note, the state time course is also known as the 'Viterbi path'.

from osl_dynamics.inference import modes

# Load state probability
alp = pickle.load(open("hmm_notts_task_10_subj/data/alpha.pkl", "rb"))

# Take the most probably state at each time point
stc = modes.argmax_time_courses(alp)

#%%
# `stc` is a list of numpy arrays, one for each subject. Let's plot the state time course for the first subject to get a feel for what it looks like.

plotting.plot_alpha(stc[0], n_samples=2000)

#%%
# We see for this particular subject, state 6 is activated a lot at the start of the time series. We also see short activations of the other states.
#
# State power maps
# ****************
# Another thing we could to do to get a feel for the HMM fit is plot the state power maps, see the `HMM Power Analysis tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_power_analysis.html>`_ for more details on how this is calculated.

from osl_dynamics.analysis import power

# Load state spectra
f = np.load("hmm_notts_task_10_subj/data/f.npy")
psd = np.load("hmm_notts_task_10_subj/data/psd.npy")

# Calculate power (integrating over all frequencies)
p = power.variance_from_spectra(f, psd)

# Average over subjects
p = np.mean(p, axis=0)

# Plot (takes a few seconds for the plots to appear)
fig, ax = power.save(
    p,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
)

#%%
# These power maps show recognisable networks which gives us confidence in the HMM fit. We can also see the first state resembles a visual network, which we expect will be involved in the task.
#
# Aligning the state time course to the event timings
# ***************************************************
# When we trained the HMM, we prepared the data using time-delay embedding and principal component analysis. As explained in the `HMM Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_training_real_data.html>`_ when we do this, we lose a few time points from each subject. This means the event timings in `event_indices` are slightly off. To prepare the data we used `n_embedding=15`, which meant 7 data points are lost from the start and end of each subject's time series. To account for this we simply need to subtract 7 from the event timings. Let's do this.

# Account for losing 7 data points due to time-delay embedding
event_indices_tde = [v - 7 for v in event_indices]

# Subtracting 7 might lead to the first event occuring as a negative sample index
# This corresponds to an event we chopped off when we time-delay embedded
#
# Remove events that we missed due to time-delay embedding
event_indices_tde = [v[v > 0] for v in event_indices_tde]

#%%
# We also lose a few data points from the end of each subjects time series due to us separating it into sequences. Let's remove events that we cut off when we separated the training data into sequences.

# Remove events that we missed due separating into sequences
event_indices_tde = [v[v < s.shape[0]] for v, s in zip(event_indices_tde, stc)]

#%%
# Epoch the state time course
# ***************************
# Now we have trimmed the event timings to match the state time course, let's epoch the state time courses.

# Epoch around events
stc_epochs = []
for s, v in zip(stc, event_indices_tde):
    stc_epochs.append(task.epoch(s, v, pre=250, post=1000))
    
# Concatenate over subjects
concat_stc_epochs = np.concatenate(stc_epochs)
print(concat_stc_epochs.shape)

# Average over epochs
avg_stc_epoch = np.nanmean(concat_stc_epochs, axis=0)
print(avg_stc_epoch.shape)

#%%
# Let's plot the epoched state time course averaged over epochs.

n_states = avg_stc_epoch.shape[1]  # number of states
t = (np.arange(avg_stc_epoch.shape[0]) - 250) / fs  # time axis

# Plot the visual task
fig, ax = plotting.plot_line(
    [t] * n_states,
    avg_stc_epoch.T,
    x_range=[-1, 4],
    x_label="Time (s)",
    y_label="Average State Activation",
)
ax.axvline(color="r", linestyle="--")

#%%
# It's not clear from this plot what the response to the event was. This is because each state has a different average activation right before the event. To see the effect of the event more clearly, it's common to 'baseline correct' the average state activation over the period before the event. Let's do this.

# Calculate the baseline
pre = 250  # number of samples before the event
base_corr = np.nanmean(avg_stc_epoch[:pre], axis=0, keepdims=True)

# Remove it from the epoched state time courses
corr_avg_stc_epoch = avg_stc_epoch - base_corr

# Plot the visual task
fig, ax = plotting.plot_line(
    [t] * n_states,
    corr_avg_stc_epoch.T,
    labels=[f"State {i}" for i in range(1, n_states + 1)],
    x_range=[-1, 4],
    x_label="Time (s)",
    y_label="Average State Activation",
)
ax.axvline(color="r", linestyle="--")

#%%
# The visual response is much cleaner. We also see it is the first state, which is the visual network, that activates in response to the task. This plot allows us to understand the dynamics of the task, i.e. when the network occurs and for how long the it lasts to a very high temporal precision. E.g. we can see the visual response occurs approximately 100 ms after the event and that it is very short lived.
#
# Subject-specific evoked responses
# *********************************
# When we see an evoked response we should test if it's statistically significant. We want to see if the peak in the group averaged epoched state time course is significantly greater than zero. Before we do that, we need to calculate the subject-specific epoched state time courses.

# We already have the trial specific epochs separated by subjects in stc_epochs,
# we just need to average each subject's trials separately
subj_stc_epochs = []
for epochs in stc_epochs:
    subj_stc_epochs.append(np.nanmean(epochs, axis=0))
subj_stc_epochs = np.array(subj_stc_epochs)
print(subj_stc_epochs.shape)

# Baseline correct using the samples before the event
subj_stc_epochs -= np.mean(subj_stc_epochs[:, :pre], axis=1, keepdims=True)

#%%
# We can see now we have a (subjects, time, states) array. Let's plot the evoked response of state 1 for each subject.

n_subjects = subj_stc_epochs.shape[0]
plotting.plot_line(
    [t] * n_subjects,
    subj_stc_epochs[:, :, 0],
    labels=[f"Subject {i}" for i in range(1, n_subjects + 1)],
    x_label="Time (s)",
    y_label="Average State Activation",
)

#%%
# We see there is a lot of variability between subjects but we consistently see a similar response to the task.
#
# Statistical significance testing
# ********************************
# To test for significance we will use a sign flipping permutation test and correct for multiple comparisons using the maximum statistic. osl-dynamics has a function for doing statistical significance testing on evoked responses: `analysis.statistics.evoked_response_max_stat_perm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/statistics/index.html#osl_dynamics.analysis.statistics.evoked_response_max_stat_perm>`_. Let's use this function.

from osl_dynamics.analysis import statistics

# Calculate p-values
pvalues = statistics.evoked_response_max_stat_perm(subj_stc_epochs, n_perm=100)
print(pvalues.shape)

# Do any time points/states come out as significant?
print("Number of time points with at least 1 state with p-value < 0.05:", np.sum(np.any(pvalues < 0.05, axis=-1)))

#%%
# Let's plot the significant time points and states.

plotting.plot_evoked_response(
    t,
    np.mean(subj_stc_epochs, axis=0),
    pvalues,
    labels=[f"State {i + 1}" for i in range(subj_stc_epochs.shape[-1])],
    significance_level=0.05,
    x_label="Time (s)",
    y_label="Average State Activation",
)

#%%
# We can see the visual state shows a statistically significant response to the task.
#
# Wrap Up
# ^^^^^^^
# - We've shown how to epoch around events and use an HMM to understand how brain network dynamics respond to a visual task.
