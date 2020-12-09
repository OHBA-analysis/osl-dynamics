"""Example script for running inference on simulated HSMM data.

- Should achieve a dice coefficient of ~0.99.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data
from vrad.inference import metrics, states, tf_ops
from vrad.models import RNNGaussian
from vrad.simulation import HSMMSimulation
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2
gamma_shape = 10
gamma_scale = 5

n_states = 5
sequence_length = 400
batch_size = 32

do_annealing = True
annealing_sharpness = 10

n_epochs = 300
n_epochs_annealing = 100

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = None

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 96

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_covariances = True

alpha_xform = "softmax"
alpha_temperature = 0.15
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Load covariances for each state
example_file_directory = Path(__file__).parent / "files"
cov = np.load(example_file_directory / "hmm_cov.npy")

# Simulate data
print("Simulating data")
sim = HSMMSimulation(
    n_samples=n_samples,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    zero_means=True,
    covariances=cov,
    observation_error=observation_error,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim)
n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_covariances=learn_covariances,
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
)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=n_epochs)

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
inf_alpha = model.predict_states(prediction_dataset)[0]
inf_stc = states.time_courses(inf_alpha)
sim_stc = sim.state_time_course

sim_stc, inf_stc = states.match_states(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Sample from the model RNN
mod_alpha = model.sample_alpha(n_samples=20000)
mod_stc = states.time_courses(mod_alpha)

# Plot lifetime distributions for the ground truth, inferred state time course
# and sampled state time course
plotting.plot_state_lifetimes(sim_stc, filename="sim_lt.png")
plotting.plot_state_lifetimes(inf_stc, filename="inf_lt.png")
plotting.plot_state_lifetimes(mod_stc, filename="mod_lt.png")

# Delete the temporary folder holding the data
meg_data.delete_dir()
