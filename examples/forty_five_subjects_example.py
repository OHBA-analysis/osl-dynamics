"""Example script for running inference on resting-state MEG data (forty five subjects).

- The data is stored on the BMRC cluster: /well/woolrich/projects/uk_meg_notts
- Uses the final covariances inferred by an HMM fit from OSL for the covariance of each
  state.
- Covariances are NOT trainable.
- Achieves a dice coefficient of ~0.78 (when compared to the OSL HMM state time course).
- Achieves a free energy of ~8,100,000.
"""

print("Setting up")
import numpy as np
from vrad.data import Data
from vrad.analysis import maps, spectral
from vrad.inference import metrics, states, tf_ops
from vrad.models import RIGO

# GPU settings
tf_ops.gpu_growth()
multi_gpu = True

# Settings
n_states = 6
sequence_length = 400
batch_size = 128

do_annealing = True
annealing_sharpness = 10

n_epochs = 50
n_epochs_annealing = 20

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 64

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_covariances = False

alpha_xform = "categorical"
alpha_temperature = 1.0
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Read MEG data
print("Reading MEG data")
prepared_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/prepared_data/subject{i}.mat"
        for i in range(1, 46)
    ],
    sampling_frequency=250,
    n_embeddings=15,
    n_pca_components=80,
    whiten=False,
    prepared=True,
)
n_channels = prepared_data.n_channels

# Prepare dataset
training_dataset = prepared_data.training_dataset(sequence_length, batch_size)
prediction_dataset = prepared_data.prediction_dataset(sequence_length, batch_size)

# Initialise covariances with final HMM covariances
hmm = data.OSL_HMM("/well/woolrich/projects/uk_meg_notts/eo/nSubjects-45_K-6/hmm.mat")
initial_covariances = hmm.covariances

# Build model
model = RIGO(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_covariances=learn_covariances,
    initial_covariances=initial_covariances,
    rnn_type=rnn_type,
    rnn_normalization=rnn_normalization,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    theta_normalization=theta_normalization,
    alpha_xform=alpha_xform,
    alpha_temperature=alpha_temperature,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    learning_rate=learning_rate,
    multi_gpu=multi_gpu,
)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=n_epochs,
    save_best_after=n_epochs_annealing,
    save_filepath="tmp/model",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time courses
alpha = model.predict_states(prediction_dataset)
inf_stc = np.concatenate(states.time_courses(alpha), axis=0)
hmm_stc = np.concatenate(
    data.manipulation.trim_time_series(
        time_series=hmm.state_time_course,
        sequence_length=sequence_length,
        discontinuities=hmm.discontinuities,
    ),
    axis=0,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))

# Load preprocessed data to calculate spatial power maps
preprocessed_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/preproc_data/subject{i}.mat"
        for i in range(1, 46)
    ]
)
preprocessed_time_series = preprocessed_data.trim_raw_time_series(
    sequence_length=sequence_length,
    n_embeddings=prepared_data.n_emebddings,
)

# Compute spectra for states
f, psd, coh = spectral.multitaper_spectra(
    data=preprocessed_time_series,
    alpha=alpha,
    sampling_frequency=prepared_data.sampling_frequency,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
)

# Perform spectral decomposition (into 2 components) based on coherence spectra
components = spectral.decompose_spectra(coh, n_components=2)

# Calculate spatial maps
p_map, c_map = maps.state_power_maps(f, psd, components)

# Save the power map for the first component as NIFTI file
# (The second component is noise)
maps.save_nii_file(
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    power_map=p_map,
    filename="power_maps.nii.gz",
    component=0,
    subtract_mean=True,
)

# Delete the temporary folder holding the data
prepared_data.delete_dir()
preprocessed_data.delete_dir()
