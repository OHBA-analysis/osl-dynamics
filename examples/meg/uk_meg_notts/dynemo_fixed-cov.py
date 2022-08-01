"""Example script for running inference on resting-state MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/projects/uk_meg_notts
- Uses the final covariances inferred by an HMM fit from OSL for the covariance of each
  mode.
- Covariances are NOT trainable.
- Achieves a dice coefficient of ~0.94 (when compared to the OSL HMM mode time course).
"""

print("Setting up")
from osl_dynamics.data import OSL_HMM, Data, processing
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.dynemo import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=6,
    n_channels=80,
    sequence_length=400,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=False,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=64,
    learning_rate=0.01,
    n_epochs=200,
)

# Read MEG data
print("Reading MEG data")
prepared_data = Data(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/prepared_data/subject1.mat",
    sampling_frequency=250,
    n_embeddings=15,
)

# Prepare dataset
training_dataset = prepared_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)
prediction_dataset = prepared_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Initialise covariances with the final HMM covariances
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-1_K-6/hmm.mat"
)
config.initial_covariances = hmm.covariances

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/model",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time courses
alpha = model.get_alpha(prediction_dataset)
inf_stc = modes.argmax_time_courses(alpha)
hmm_stc = processing.trim_time_series(
    time_series=hmm.mode_time_course(),
    sequence_length=config.sequence_length,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))
