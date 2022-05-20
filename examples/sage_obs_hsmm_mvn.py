"""Example script for running inference on simulated HMM-MVN data.

- Should achieve a dice coefficient of ~0.98.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.sage import Config, SAGE
from osl_dynamics.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=15,
    sequence_length=200,
    inference_n_units=64,
    inference_n_layers=2,
    inference_normalization="layer",
    model_n_units=32,
    model_normalization="layer",
    des_n_units=32,
    des_normalization="layer",
    learn_means=True,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=300,
)

# Mixtures of modes to include in the simulation
mixed_state_vectors = np.array(
    [[0.5, 0.5, 0, 0, 0], [0, 0.3, 0, 0.7, 0], [0, 0, 0.6, 0.4, 0]]
)

# Simulate data
print("Simulating data")
sim = simulation.MixedHSMM_MVN(
    n_samples=25600,
    n_channels=config.n_channels,
    mixed_state_vectors=mixed_state_vectors,
    gamma_shape=20,
    gamma_scale=5,
    means="random",
    covariances="random",
    random_seed=123,
)
sim.standardize()
input_data = data.Data(sim.time_series)


# Prepare dataset
training_dataset = input_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)

prediction_dataset = input_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = SAGE(config)

print("Training model")
history = model.train(training_dataset)

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_stc = sim.mode_time_course
sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (Sage):", modes.fractional_occupancies(inf_stc))

# Inferred covariances
covariances = model.get_covariances()
plotting.plot_matrices(covariances, filename="cov_inf.png")
plotting.plot_matrices(sim.covariances, filename="cov_sim.png")