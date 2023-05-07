"""Example script for fitting an amplitude envelope HMM to the Nottingham UK MEG partnership data.

"""

print("Setting up")
import os
import pickle
import numpy as np

from osl_dynamics.analysis import power, connectivity
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting

#%% Settings

train_model = False

# Create directories to hold plots, the trained model and post-hoc analysis
plots_dir = "results/plots"
model_dir = "results/model"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Config
config = Config(
    n_states=8,
    n_channels=42,
    sequence_length=2000,
    learn_means=True,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=32,
    learning_rate=1e-3,
    n_epochs=20,
)

#%% Training dataset

# Directory containing source reconstructed data
src_data_dir = "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec"

# Load source reconstructured data
training_data = Data(
    [src_data_dir + f"/subject{i}.mat" for i in range(1, 56)],
    sampling_frequency=250,
)

# Prepare the data: time-delay embedding, PCA and standardization
training_data.prepare(amplitude_envelope=True, n_window=6)

#%% Model training

if train_model:
    # Build model
    model = Model(config)
    model.summary()

    # Initialization: train the HMM a few times with a randomly sampled state
    # time course used to calculate the initial observation model parameters
    model.random_state_time_course_initialization(training_data, n_epochs=2, n_init=3)

    # Main training
    history = model.fit(training_data)

    # Save
    model.save(model_dir)
    pickle.dump(history, open(model_dir + "/history.pkl", "wb"))

else:
    model = Model.load(model_dir)

#%% Inferred parameters

# Get subject-specific inferred state probabilities
alpha = model.get_alpha(training_data)

# Plot the alphas for the first subject
plotting.plot_alpha(alpha[0], n_samples=2000, filename=plots_dir + "/alpha.png")

# Get the inferred state transition probability matrix
trans_prob = model.get_trans_prob()
np.fill_diagonal(trans_prob, 0)

plotting.plot_matrices(trans_prob, filename=plots_dir + "/trans_prob.png")

# Inferred means and covariances
means, covs = model.get_means_covariances()

plotting.plot_matrices(covs, filename=plots_dir + "/covs.png")

#%% Summary statistics

# Calculate a state time course from the state probabilities
stc = modes.argmax_time_courses(alpha, concatenate=True)

# Fractional occupancies
fo = modes.fractional_occupancies(stc)

print("Fractional occupancies:", fo)

# Lifetimes
lt = modes.mean_lifetimes(stc, sampling_frequency=training_data.sampling_frequency)

print("Lifetimes:", lt)

# Intervals
intv = modes.mean_intervals(stc, sampling_frequency=training_data.sampling_frequency)

print("Intervals:", intv)

#%% Mean activity and connectivity maps

# Source reconstruction files needed for plotting maps
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Plot mean activity maps
power.save(
    power_map=means,
    filename=plots_dir + "/mean_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
)

# Plot correlation maps
conn_map = abs(covs)
conn_map = connectivity.threshold(conn_map, percentile=95)
connectivity.save(
    connectivity_map=conn_map,
    filename=plots_dir + "/corr_.png",
    parcellation_file=parcellation_file,
)

#%% Clean up

# Delete temporary directory
training_data.delete_dir()
