"""Example script for running inference on data with a mixture of HSMM states.

- Demonstrates VRAD's ability to infer a soft mixture of states.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data
from vrad.inference import states, tf_ops, metrics
from vrad.models import RNNGaussian
from vrad.simulation import MixedHSMMSimulation
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2
gamma_shape = 20
gamma_scale = 10

n_states = 5
sequence_length = 200
batch_size = 16

do_annealing = True
annealing_sharpness = 10

n_epochs = 200
n_epochs_annealing = 100

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = None

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 64

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_covariances = True

alpha_xform = "softmax"
alpha_temperature = 0.2
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Load covariances for each state
example_file_directory = Path(__file__).parent / "files"
cov = np.load(example_file_directory / "hmm_cov.npy")

# Mixtures of states to include in the simulation
mixed_state_vectors = np.array(
    [[0.5, 0.5, 0, 0, 0], [0, 0.3, 0, 0.7, 0], [0, 0, 0.6, 0.4, 0]]
)

# Simulate data
print("Simulating data")
sim = MixedHSMMSimulation(
    n_samples=n_samples,
    mixed_state_vectors=mixed_state_vectors,
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
history = model.fit(
    training_dataset,
    epochs=n_epochs,
    save_best_after=n_epochs_annealing,
    save_filepath="tmp/weights",
)

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Compare the inferred state time course to the ground truth
alpha = model.predict_states(prediction_dataset)[0]
matched_sim_stc, matched_alpha = states.match_states(sim.state_time_course, alpha)
plotting.plot_state_time_courses(
    matched_alpha, matched_sim_stc, n_samples=10000, filename="stc.png"
)

corr = metrics.correlation(matched_alpha, matched_sim_stc)
print("Correlation (VRAD vs Simulation):", corr)

# Delete the temporary folder holding the data
meg_data.delete_dir()
