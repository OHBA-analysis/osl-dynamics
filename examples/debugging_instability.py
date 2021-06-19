"""Example script demonstrating a training instability.

- If you have a large number of units, e.g. 256 or 512 there is a training
  instability where the KL loss suddenly goes to inf.
    - This arises from the posterior/prior mean and variance entering the KL
      divergence calculation being nan.
    - I.e. at some point the posterior/prior means and variance turns to nan
      and this causes the KL loss to go to inf.
- Instability occurs at:
    - ~2 epochs with a learning rate of 0.01.
    - ~5 epochs with a learning rate of 0.005 the KL loss blows up (but loss
      does not go to nan).
    - There is some run-to-run variability. The instability doesn't always
      occur. This could indicate this occurs due to the RNN weight initialisation.
- Setting gradient_clip=0.5 seems to fix the instability.
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
    inference_n_layers=1,
    inference_n_units=512,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_layers=1,
    model_n_units=512,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=False,
    initial_alpha_temperature=1.0,
    learn_covariances=False,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
    batch_size=32,
    learning_rate=0.01,
    gradient_clip=None,
    n_epochs=50,
    multi_gpu=False,
)

# Read MEG data
print("Reading MEG data")
prepared_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/prepared_data/subject{i}.mat"
        for i in range(1, 46)
    ],
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
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/results/nSubjects-45_K-6/hmm.mat"
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
inf_stc = states.time_courses(alpha, concatenate=True)
hmm_stc = manipulation.trim_time_series(
    time_series=hmm.state_time_course(),
    sequence_length=config.sequence_length,
    concatenate=True,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))

# Delete the temporary folder holding the data
prepared_data.delete_dir()
