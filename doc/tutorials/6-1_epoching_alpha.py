"""
Epoching: Network Response
==========================

In this tutorial we will cover how to epoch the inferred state/mode time courses. We will use the functionality in MNE to do this. This requires us to have the continuous training data in `.fif` format.
"""

#%%
# Download source data
# ^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data from `OSF <https://osf.io/by2tc/>`_.

import os

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 88 MB)
get_data("wakeman_henson_giles_5_subjects", rename="src")

#%%
# Download inferred parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_inf_params(name, rename):
    os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 17 MB)
get_inf_params("tde_hmm_wakeman_henson_giles_5_subjects", rename="results/inf_params")

#%%
# Convert to mne.io.Raw
# ^^^^^^^^^^^^^^^^^^^^^
# To epoch the state/mode time courses, we first load the inferred state/mode time courses. These **will not include any bad segments** and will have time points removed due to time-delay embedding and separating into sequences (or other data processing steps). To handle this we will make use of MNE by converting the state/mode time courses to mne.io.Raw format then we will use the usual MNE functions to epoch.

import os
import pickle
from osl_dynamics.inference import modes

# Load inferred state/mode time courses
alp = pickle.load(open("results/inf_params/alp.pkl", "rb"))
alp = alp[0]

# Corresponding fif file containing the parcellated data
parc_fif = "src/sub01_run01/sflip_parc-raw.fif"

# Convert the state/mode time courses to mne.io.Raw format
alp_raw = modes.convert_to_mne_raw(alp, parc_fif, n_embeddings=15)

# Save
os.makedirs("results/alphas", exist_ok=True)
alp_raw.save("results/alphas/sub01_run01_alpha-raw.fif", overwrite=True)

#%%
# Note:
# - If the parcellated data fif file contains a 'stim' channel, it will automatically be transferred to the `alp_raw` object. This is useful because the stim channel will be used to find events.
# - If you used time-delay embedding or a moving average when you prepared the data, then you need to pass `n_embeddings` or `n_window` respectively to `convert_to_mne_raw` to correctly account for the time points that are trimmed from the data.
#
# Get the epoched data
# ^^^^^^^^^^^^^^^^^^^^
# To get the epoched data (trials) we use the standard MNE approach by first finding the events then epoching.

import mne

# Find events
events = mne.find_events(alp_raw, min_duration=0.005)

# Trigger IDs for events in the task
event_ids = {"famous": 6, "unfamiliar": 13, "scrambled": 17}

# Pick the channels are correspond to the network time courses
alp_raw.pick("misc")

# Epoch
epochs = mne.Epochs(
    alp_raw,
    events,
    event_ids,
    tmin=-0.1,
    tmax=1.0,
    baseline=None,
)

# Get the epoched data
data = epochs.get_data()
print(data.shape)

# Time axis
t = epochs.times

#%%
# We see `data` is (trials, time, states/modes). This is the trials for this session. Note, when we do `.get_data()` MNE will automatically drop any epoch that overlaps with a bad segment in the parcellated data fif file.
#
# If we want a particular trial type, then we can get it using the normal MNE approach:

famous = epochs["famous"].get_data()
print(famous.shape)

#%%
# Statistical significance testing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can average the trials (epochs) for a given subject and do stats to see if there's a significant response. See the `Network Response <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/6-2_group_network_response.html>`_ tutorial for how to do group-level stats on the subject-level network responses.
