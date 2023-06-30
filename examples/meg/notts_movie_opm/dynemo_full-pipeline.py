"""Train DyNeMo on Nottingham movie data.

"""

print("Setting up")
import pickle
import numpy as np
import os.path as op
from os import makedirs
from osl_dynamics import analysis, data, inference
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.utils import plotting

#%% Settings

# GPU settings
inference.tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=6,
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
    batch_size=32,
    learning_rate=0.0025,
    gradient_clip=0.5,
    n_epochs=400,
    multi_gpu=False,
)

subjects_dir = '/Users/woolrich/homedir/vols_data/notts_movie_opm'
#subjects_dir = '/well/woolrich/projects/notts_movie_opm'

n_subject_init_runs = 5
run = 1

use_pre_trained_model = True

subjects_to_do = np.arange(0, 10)
sessions_to_do = np.arange(0, 2)
subj_sess_2exclude = np.zeros([10, 2]).astype(bool)

subj_sess_2exclude = np.ones([10, 2]).astype(bool)
subj_sess_2exclude[0:2,:] = False

#%% Setup filenames

subjects = []
sf_files = []
recon_dir = op.join(subjects_dir, 'recon')
for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:
            sub_dir = 'sub-' + ('{}'.format(subjects_to_do[sub]+1)).zfill(3)
            ses_dir = 'ses-' + ('{}'.format(sessions_to_do[ses]+1)).zfill(3)
            subject = sub_dir + '_' + ses_dir
            sf_file = op.join(recon_dir, subject + '/sflip_parc.npy')
            subjects.append(subject)
            sf_files.append(sf_file)

# Directories
output_id = f"dynemo_run{run}"

model_dir = f"{recon_dir}/{output_id}/model"
analysis_dir = f"{recon_dir}/{output_id}/analysis"
maps_dir = f"{recon_dir}/{output_id}/maps"
tmp_dir = f"{recon_dir}/{output_id}/tmp"

makedirs(model_dir, exist_ok=True)
makedirs(analysis_dir, exist_ok=True)
makedirs(maps_dir, exist_ok=True)

#%% Training dataset

# Prepare the data for training
training_data = data.Data(sf_files)
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": config.n_channels},
    "standardize": {},
}
training_data.prepare(methods)

#%% Build model

model = Model(config)
model.summary()

#%% Model training

if not use_pre_trained_model:
    # Multi-start initialization
    init_history = model.multistart_initialization(training_data, n_epochs=20, n_init=10)

    with open(f"{model_dir}/init_history.pkl", "wb") as file:
        pickle.dump(init_history, file)

    # Main training on the full dataset
    print("Training final model")
    history = model.fit(
        training_data,
        save_best_after=config.n_kl_annealing_epochs,
        save_filepath=f"{model_dir}/weights",
    )

    with open(f"{model_dir}/history.pkl", "wb") as file:
        pickle.dump(history, file)

else:
    # Load a pre-trained model
    model.load_weights(f"{model_dir}/weights")

    with open(f"{model_dir}/history.pkl", "rb") as file:
        history = pickle.load(file)

# Training loss
ll_loss, kl_loss = model.losses(training_data)
loss = ll_loss + kl_loss
print(f"training loss: {loss}")

with open(f"{model_dir}/loss.dat", "w") as file:
    file.write(f"training loss = {loss}\n")

plotting.plot_line(
    [range(config.n_epochs)],
    [history["loss"]],
    x_label="Epoch",
    y_label="Loss",
    filename=f"{model_dir}/loss.png",
)

#%% Mode analysis

# Alpha time course for each subject
a = model.get_alpha(training_data)

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

#%% Spectral analysis

# Source reconstructed data
src_rec_data = training_data.trim_time_series(
    sequence_length=config.sequence_length,
    n_embeddings=training_data.n_embeddings,
    prepared=False,
)

# Calculate subject-specific mode PSDs and coherences
f, psd, coh, w = analysis.spectral.regression_spectra(
    data=src_rec_data,
    alpha=a,
    sampling_frequency=250,
    window_length=1000,
    frequency_range=[1, 45],
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

#%% Power and connectivity maps

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Calculate relative (regression coefficients only) power maps using the
# mode PSDs
power_map = analysis.power.variance_from_spectra(f, gpsd[0])
analysis.power.save(
    power_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=f"{maps_dir}/pow_.png",
    subtract_mean=True,
)

# Calculate connectivity maps using mode coherences
coh_map = analysis.connectivity.mean_coherence_from_spectra(f, gcoh)

# Use a GMM to threshold the connectivity maps
coh_map = analysis.connectivity.gmm_threshold(
    coh_map,
    subtract_mean=True,
    standardize=True,
    one_component_percentile=95,
    n_sigma=2,
    filename=f"{maps_dir}/gmm_coh_.png",
    plot_kwargs={
        "x_label": "Standardised Relative Coherence",
        "y_label": "Probability",
    },
)

# Save coherence networks
analysis.connectivity.save(
    coh_map,
    filename=f"{maps_dir}/coh_.png",
    parcellation_file=parcellation_file,
)

#%% Clean up

training_data.delete_dir()
