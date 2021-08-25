"""Example script for running inference on resting-state MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/projects/uk_meg_notts
- Uses the final covariances inferred by an HMM fit from OSL for the covariance of each
  state.
- Covariances are NOT trainable.
- Achieves a dice coefficient of ~0.94 (when compared to the OSL HMM state time course).
"""

print("Setting up")
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
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization="layer",
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
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

config.n_channels = prepared_data.n_channels

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

# Inferred state mixing factors and state time courses
alpha = model.predict_states(prediction_dataset)
inf_stc = states.time_courses(alpha)
hmm_stc = manipulation.trim_time_series(
    time_series=hmm.state_time_course(),
    sequence_length=config.sequence_length,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))

# Delete the temporary folder holding the data
prepared_data.delete_dir()
