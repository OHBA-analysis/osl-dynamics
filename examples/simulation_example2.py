"""Example script for running inference on simulated HSMM data.

- Should achieve a dice coefficient of ~0.99.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from vrad import data
from vrad.inference import gmm, metrics, states, tf_ops
from vrad.models import RNNGaussian
from vrad.simulation import HSMMSimulation
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25000
observation_error = 0.2
gamma_shape = 10
gamma_scale = 5

n_states = 5
sequence_length = 200
batch_size = 32

do_annealing = True
annealing_sharpness = 5

n_epochs = 500
n_epochs_annealing = 150

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 32
n_units_model = 48

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Load covariances for each state
cov = np.load("files/hmm_cov.npy")

# Simulate data
print("Simulating data")
sim = HSMMSimulation(
    n_samples=n_samples,
    n_states=n_states,
    sim_varying_means=learn_means,
    covariances=cov,
    observation_error=observation_error,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim)
n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Initialisation of means and covariances
initial_means, initial_covariances = gmm.final_means_covariances(
    meg_data.subjects[0],
    n_states,
    gmm_kwargs={
        "n_init": 1,
        "verbose": 2,
        "verbose_interval": 50,
        "max_iter": 10000,
        "tol": 1e-6,
    },
    retry_attempts=1,
    learn_means=False,
    random_seed=124,
)

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_means=initial_means,
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
    save_filepath="/well/woolrich/shared/vrad/trained_models/simulation_example2/weights",
)

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
alpha = model.predict_states(prediction_dataset)[0]
inf_stc = states.time_courses(alpha)

# Find correspondance to ground truth state time courses
matched_sim_stc, matched_inf_stc = states.match_states(sim.state_time_course, inf_stc)

print("Dice coefficient:", metrics.dice_coefficient(matched_sim_stc, matched_inf_stc))

# Sample from the model RNN
mod_stc = model.sample_state_time_course(n_samples=25000)

# Plot lifetime distributions for the ground truth, inferred state time course
# and sampled state time course
plotting.plot_state_lifetimes(sim.state_time_course, filename="sim_lt.png")
plotting.plot_state_lifetimes(inf_stc, filename="inf_lt.png")
plotting.plot_state_lifetimes(mod_stc, filename="mod_lt.png")

# Delete the temporary folder holding the data
meg_data.delete_dir()
