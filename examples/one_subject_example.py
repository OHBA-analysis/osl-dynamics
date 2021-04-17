"""Example script for running inference on resting-state MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/projects/uk_meg_notts
- Uses the final covariances inferred by an HMM fit from OSL for the covariance of each
  state.
- Covariances are NOT trainable.
- Achieves a dice coefficient of ~0.92 (when compared to the OSL HMM state time course).
"""

print("Setting up")
from vrad.analysis import maps, spectral
from vrad.data import OSL_HMM, Data, manipulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=6,
    sequence_length=400,
    inference_rnn="lstm",
    inference_n_layers=1,
    inference_n_units=64,
    inference_normalization="layer",
    inference_activation="elu",
    inference_dropout_rate=0.0,
    model_rnn="lstm",
    model_n_layers=1,
    model_n_units=64,
    model_normalization="layer",
    model_activation="elu",
    model_dropout_rate=0.0,
    theta_normalization="layer",
    alpha_xform="gumbel-softmax",
    learn_alpha_temperature=False,
    initial_alpha_temperature=1.0,
    learn_covariances=False,
    learn_alpha_scaling=False,
    normalize_covariances=False,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_epochs_kl_annealing=100,
    batch_size=64,
    learning_rate=0.01,
    n_epochs=200,
    multi_gpu=True,
)

# Read MEG data
print("Reading MEG data")
prepared_data = Data(
    "/well/woolrich/projects/uk_meg_notts/eo/prepared_data/subject1.mat",
    sampling_frequency=250,
    n_embeddings=15,
)

config.n_channels = prepared_data.n_channels

# Prepare dataset
training_dataset = prepared_data.training_dataset(
    config.sequence_length, config.batch_size
)
prediction_dataset = prepared_data.prediction_dataset(
    config.sequence_length, config.batch_size
)

# Initialise covariances with the final HMM covariances
hmm = OSL_HMM("/well/woolrich/projects/uk_meg_notts/eo/results/nSubjects-1_K-6/hmm.mat")
config.initial_covariances = hmm.covariances

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_epochs_kl_annealing,
    save_filepath="tmp/model",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time courses
alpha = model.predict_states(prediction_dataset)
inf_stc = states.time_courses(alpha)
hmm_stc = manipulation.trim_time_series(
    time_series=hmm.state_time_course(),
    sequence_length=config.sequence_length,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))

# Load preprocessed data to calculate spatial power maps
preprocessed_data = Data(
    "/well/woolrich/projects/uk_meg_notts/eo/preproc_data/subject1.mat",
)
preprocessed_time_series = preprocessed_data.trim_raw_time_series(
    sequence_length=config.sequence_length,
    n_embeddings=prepared_data.n_embeddings,
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

# Calculate spatial power maps
p_map = maps.state_power_maps(f, psd, components)

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
