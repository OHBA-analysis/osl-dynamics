"""Example script for fitting an HMM to the Nottingham UK MEG partnership data.

In this script we reproduce the wideband power/coherence maps presented
in Diego Vidaurre's 2018 Nature Communications paper.
"""

print("Setting up")
import os
import pickle
import numpy as np
from osl_dynamics.analysis import spectral, power, connectivity
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting

# --------
# Settings

train_model = True
calc_spectra = True

# Create directories to hold plots, the trained model and post-hoc analysis
plots_dir = "results/plots"
model_dir = "results/model"
spectra_dir = "results/spectra"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(spectra_dir, exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Config
config = Config(
    n_states=12,
    n_channels=80,
    sequence_length=2000,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=32,
    learning_rate=1e-3,
    n_epochs=15,
)

# ----------------
# Training dataset

# Directory containing source reconstructed data
src_data_dir = "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec"

# Load source reconstructured data
training_data = Data(
    [src_data_dir + f"/subject{i}.mat" for i in range(1, 56)],
    sampling_frequency=250,
)

# Prepare the data: time-delay embedding, PCA and standardization
training_data.prepare(n_embeddings=15, n_pca_components=config.n_channels)

# --------------
# Model training

# Build model
model = Model(config)
model.summary()

if train_model:
    # Initialization: train the HMM a few times with a randomly sampled state
    # time course used to calculate the initial observation model parameters
    model.random_state_time_course_initialization(training_data, n_epochs=2, n_init=3)

    # Main training
    history = model.fit(training_data)

    # Save
    model.save(model_dir)
    pickle.dump(history, open(model_dir + "/history.pkl", "wb"))

else:
    model.load(model_dir)

# -------------------
# Inferred parameters

# Get subject-specific inferred state probabilities
alpha = model.get_alpha(training_data)

# Plot the alphas for the first subject
plotting.plot_alpha(alpha[0], n_samples=2000, filename=plots_dir + "/alpha.png")

# Get the inferred state transition probability matrix
trans_prob = model.get_trans_prob()
np.fill_diagonal(trans_prob, 0)

plotting.plot_matrices(trans_prob, filename=plots_dir + "/trans_prob.png")

# Inferred covariances
covs = model.get_covariances()

plotting.plot_matrices(covs, filename=plots_dir + "/covs.png")

# ------------------
# Summary statistics

# Calculate a state time course from the state probabilities
stc = modes.argmax_time_courses(alpha, concatenate=True)

# Fractional occupancies
fo = modes.fractional_occupancies(stc)

print("Fractional occupancies:", fo)

# Lifetimes
mean_lt, _ = modes.lifetime_statistics(
    stc, sampling_frequency=training_data.sampling_frequency
)

print("Lifetimes:", mean_lt)

# Intervals
intv = modes.intervals(stc, sampling_frequency=training_data.sampling_frequency)
mean_intv = np.array([np.mean(i) for i in intv])

print("Intervals:", mean_intv)

# -----------------
# Spectral analysis

if calc_spectra:
    # Get subject-specific source reconstructed data
    data = model.get_training_time_series(training_data, prepared=False)

    # Calculate state spectra with a multitaper
    f, psd, coh, w = spectral.multitaper_spectra(
        data=data,
        alpha=alpha,
        sampling_frequency=training_data.sampling_frequency,
        time_half_bandwidth=4,
        n_tapers=7,
        frequency_range=[1, 45],
        return_weights=True,
        n_jobs=16,
    )

    # Save
    np.save(spectra_dir + "/f.npy", f)
    np.save(spectra_dir + "/psd.npy", psd)
    np.save(spectra_dir + "/coh.npy", coh)
    np.save(spectra_dir + "/w.npy", w)

else:
    f = np.load(spectra_dir + "/f.npy")
    psd = np.load(spectra_dir + "/psd.npy")
    coh = np.load(spectra_dir + "/coh.npy")
    w = np.load(spectra_dir + "/w.npy")

# Perform non-negative matrix factorization on the coherence matrix of each
# subject to find frequency bands for coherent activity
#
# We fit two spectral components to the subject-specific coherences
wideband_components = spectral.decompose_spectra(coh, n_components=2)

plotting.plot_line([f, f], wideband_components, filename=plots_dir + "/wideband.png")

# Calculate the group average of subject-specific spectra
psd = np.average(psd, axis=0, weights=w)
coh = np.average(coh, axis=0, weights=w)

# -----------------------
# Power and coherence maps

# Source reconstruction files needed for plotting maps
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Plot power maps
power_map = power.variance_from_spectra(f, psd, wideband_components)
power.save(
    power_map=power_map,
    filename=plots_dir + "/power_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    component=0,  # just plot the first component, second is noise
)

# Plot coherence maps
conn_map = connectivity.mean_coherence_from_spectra(f, coh, wideband_components)
conn_map = connectivity.gmm_threshold(
    conn_map,
    subtract_mean=True,
    standardize=True,
    one_component_percentile=95,  # we use this percentile if we don't find two components in the distribution
    n_sigma=0,  # require the mean of the GMM components to be this number of sigmas apart
    filename=plots_dir + "/gmm_conn_.png",
)
connectivity.save(
    connectivity_map=conn_map,
    filename=plots_dir + "/conn_.png",
    parcellation_file=parcellation_file,
    component=0,  # just plot the first component, second is noise
)

# --------
# Clean up

# Delete temporary directory
training_data.delete_dir()
