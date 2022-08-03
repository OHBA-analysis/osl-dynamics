"""Full pipeline for a resting-state MEG study with DyNeMo.

"""

from sys import argv
from os import makedirs

# Command line arguments
if len(argv) != 2:
    print("Need to pass one argument, e.g. python full_pipeline.py 1")
    exit()

# ID for the run
run = int(argv[1])

print("Setting up")
import numpy as np
import pickle
from osl_dynamics import analysis, data, inference
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.utils import plotting

# -------- #
# Settings #
# -------- #
# GPU settings
inference.tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=10,
    n_channels=80,
    sequence_length=200,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=200,
    n_init=10,
    n_init_epochs=20,
    batch_size=32,
    learning_rate=0.0025,
    gradient_clip=0.5,
    n_epochs=400,
    multi_gpu=False,
)

# Output id and folders
output_id = f"run{run}"
model_dir = f"models/{output_id}"
analysis_dir = f"analysis/{output_id}"
maps_dir = f"maps/{output_id}"
tmp_dir = f"tmp_{output_id}"

makedirs(model_dir, exist_ok=True)
makedirs(analysis_dir, exist_ok=True)
makedirs(maps_dir, exist_ok=True)

# ------------------------------- #
# Training and validation dataset #
# ------------------------------- #
# Dataset containing source reconstructed data
dataset_dir = "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec"

# Load data
print("Reading MEG data")
training_data = data.Data(
    [f"{dataset_dir}/subject{i}.mat" for i in range(1, 46)],
    sampling_frequency=250,
    store_dir=f"{tmp_dir}",
)
validation_data = data.Data(
    [f"{dataset_dir}/subject{i}.mat" for i in range(46, 56)],
    store_dir=f"{tmp_dir}",
)

# Prepare the data for training
training_data.prepare(n_embeddings=15, n_pca_components=config.n_channels)
validation_data.prepare(n_embeddings=15, pca_components=training_data.pca_components)

# Create tensorflow datasets for training and evaluting the model
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
    concatenate=False,
)
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
    concatenate=False,
)
validation_dataset = validation_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
    concatenate=True,
)

# --------------------------- #
# Build the main DyNeMo model #
# --------------------------- #
model = Model(config)
model.summary()

# --------------------------------------- #
# Initialisation for the mode covariances #
# --------------------------------------- #
# Choose subjects at random
n_runs = 10
subjects_used = np.random.choice(range(len(training_dataset)), n_runs, replace=False)

# Train the model a few times and keep the best one
best_loss = np.Inf
losses = []
for subject in subjects_used:
    print("Using subject", subject, "to train initial covariances")

    # Get the dataset for this subject
    subject_dataset = training_dataset[subject]

    # Reset the model weights and train
    model.reset_weights()
    model.compile()
    history = model.fit(subject_dataset, epochs=config.n_epochs)
    loss = history.history["loss"][-1]
    losses.append(loss)
    print(f"Subject {subject} loss: {loss}")

    # Record the loss of this subject's data
    if loss < best_loss:
        best_loss = loss
        subject_chosen = subject
        best_weights = model.get_weights()

print(f"Using covariances from subject {subject_chosen}")

# Restore the best model and get the inferred covariances for initialisation
model.set_weights(best_weights)
init_cov = model.get_covariances()

# Reset model for full training
model.reset()

# Set initial covariances
model.set_covariances(init_cov, update_initializer=True)

# ------------------------- #
# Train on the full dataset #
# ------------------------- #
# Concatenate the datasets from each subject
training_dataset = data.tf.concatenate_datasets(training_dataset, shuffle=True)

print("Train with different RNN initializations")
init_history = model.initialize(
    training_dataset,
    epochs=config.n_init_epochs,
    n_init=config.n_init,
)

with open(f"{model_dir}/init_history.pkl", "wb") as file:
    pickle.dump(init_history.history, file)

print("Train final model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath=f"{model_dir}/weights",
)

with open(f"{model_dir}/history.pkl", "wb") as file:
    pickle.dump(history.history, file)

"""
# Load a pre-trained model
model.load_weights(f"{model_dir}/weights")
"""

# Training loss
train_ll_loss, train_kl_loss = model.losses(training_dataset)
train_loss = train_ll_loss + train_kl_loss
print(f"training loss: {train_loss}")

# Validation loss
val_ll_loss, val_kl_loss = model.losses(validation_dataset)
val_loss = val_ll_loss + val_kl_loss
print(f"validation loss: {val_loss}")

with open(f"{model_dir}/loss.dat", "w") as file:
    file.write(f"train_loss = {train_loss}\n")
    file.write(f"val_loss = {val_loss}\n")

# ------------- #
# Mode Analysis #
# ------------- #
# Alpha time course for each subject
a = model.get_alpha(prediction_dataset)

# Order modes with respect to mean alpha values
mean_a = np.mean(np.concatenate(a), axis=0)
order = np.argsort(mean_a)[::-1]

mean_a = mean_a[order]
a = [alp[:, order] for alp in a]

print("mean_a:", mean_a)

plotting.plot_alpha(a[0], filename=f"{analysis_dir}/a.png")

# Correlation between raw alphas
a_corr = np.corrcoef(np.concatenate(a), rowvar=False) - np.eye(config.n_modes)

plotting.plot_matrices(a_corr, filename=f"{analysis_dir}/a_corr1.png")
plotting.plot_matrices(a_corr[1:, 1:], filename=f"{analysis_dir}/a_corr2.png")

# Mode covariances
D = model.get_covariances()
D = D[order]

# Trace of mode covariances
tr_D = np.trace(D, axis1=1, axis2=2)

# Normalised weighted alpha
a_NW = [tr_D[np.newaxis, ...] * alp for alp in a]
a_NW = [alp_NW / np.sum(alp_NW, axis=1)[..., np.newaxis] for alp_NW in a_NW]

plotting.plot_alpha(a_NW[0], filename=f"{analysis_dir}/a_NW.png")

# Mean normalised weighted alpha
mean_a_NW = np.mean(np.concatenate(a_NW), axis=0)

print("mean_a_NW:", mean_a_NW)

# Create a state time course from the normalised weighted alpha
argmax_a_NW = inference.modes.argmax_time_courses(a_NW)

plotting.plot_alpha(
    argmax_a_NW[0],
    n_samples=5 * training_data.sampling_frequency,
    filename=f"{analysis_dir}/argmax_a_NW.png",
)

# State statistics
lt = inference.modes.lifetimes(
    argmax_a_NW, sampling_frequency=training_data.sampling_frequency
)
intv = inference.modes.intervals(
    argmax_a_NW, sampling_frequency=training_data.sampling_frequency
)
fo = np.array(inference.modes.fractional_occupancies(argmax_a_NW))

mean_lt = np.array([np.mean(lifetimes) for lifetimes in lt])
mean_intv = np.array([np.mean(interval) for interval in intv])
mean_fo = np.mean(fo, axis=0)

print("mean_lt:", mean_lt)
print("mean_intv:", mean_intv)
print("mean_fo:", mean_fo)

plotting.plot_violin(
    [1e3 * l for l in lt],
    y_range=[0, 200],
    x_label="Mode",
    y_label="Lifetime (ms)",
    filename=f"{analysis_dir}/lt.png",
)
plotting.plot_violin(
    intv,
    y_range=[-0.2, 10],
    x_label="Mode",
    y_label="Interval (s)",
    filename=f"{analysis_dir}/intv.png",
)
plotting.plot_violin(
    [f for f in fo.T],
    x_label="Mode",
    y_label="Fractional Occupancy",
    filename=f"{analysis_dir}/fo.png",
)

# ----------------- #
# Spectral analysis #
# ----------------- #
# Source reconstructed data
src_rec_data = training_data.trim_time_series(
    sequence_length=config.sequence_length,
    n_embeddings=training_data.n_embeddings,
    prepared=False,
)

# Caclculate subject-specific mode PSDs and coherences
f, psd, coh, w = analysis.spectral.regression_spectra(
    data=src_rec_data,
    alpha=a,
    sampling_frequency=training_data.sampling_frequency,
    window_length=1000,
    frequency_range=[0, 45],
    step_size=20,
    n_sub_windows=8,
    return_weights=True,
    return_coef_int=True,
)

# Average subject-specific PSDs to get the group-level mode PSDs
gpsd = np.average(psd, axis=0, weights=w)

# Sum regression coefficients and intercept and calculate mean over channels
P = np.sum(gpsd, axis=0)
p = np.mean(P, axis=1)
e = np.std(P, axis=1) / np.sqrt(P.shape[1])

plotting.plot_line(
    [f] * p.shape[0],
    p,
    labels=[f"Mode {i}" for i in range(1, p.shape[0] + 1)],
    errors=[p - e, p + e],
    x_range=[0, 45],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    filename=f"{analysis_dir}/psd1.png",
)

# Just plot the regression coefficients averaged over channels
P = gpsd[0]
p = np.mean(P, axis=1)
e = np.std(P, axis=1) / np.sqrt(P.shape[1])

plotting.plot_line(
    [f] * p.shape[0],
    p,
    labels=[f"Mode {i}" for i in range(1, p.shape[0] + 1)],
    errors=[p - e, p + e],
    x_range=[0, 45],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    filename=f"{analysis_dir}/psd2.png",
)

# Average subject-specific coherences to get the group-level mode coherences
gcoh = np.average(coh, axis=0, weights=w)

# --------------------------- #
# Power and connectivity maps #
# --------------------------- #
# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Calculate relative power maps using the mode PSDs
power_map = analysis.power.variance_from_spectra(f, gpsd.sum(axis=0))
analysis.power.save(
    power_map=power_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=f"{maps_dir}/power_.png",
    subtract_mean=True,
)

# Calculate connectivity maps using mode coherences
conn_map = analysis.connectivity.mean_coherence_from_spectra(f, gcoh)

# Use a GMM to threshold the connectivity maps
percentile = analysis.connectivity.fit_gmm(
    conn_map,
    min_percentile=92,
    max_percentile=98,
    subtract_mean=True,
    standardize=True,
    filename=f"{maps_dir}/gmm_conn_.png",
    plot_kwargs={
        "x_label": "Standardised Relative Coherence",
        "y_label": "Probability",
    },
)
conn_map = connectivity.threshold(conn_map, percentile, subtract_mean=True)

analysis.connectivity.save(
    connectivity_map=conn_map,
    filename=f"{maps_dir}/conn_.png",
    parcellation_file=parcellation_file,
)

# -------- #
# Clean up #
# -------- #
training_data.delete_dir()
