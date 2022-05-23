"""Example script for demonstrating Sage's ability to learn long-range dependependcies.

- An HSMM is simulated.
- The prior alpha is generated based on the inferred alpha
"""

print("Setting up")
import os
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops, modes, metrics, callbacks
from osl_dynamics.models.sage import Config, Model
from osl_dynamics.utils import plotting

# Make directory to hold plots
os.makedirs("figures_longrange", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=8,
    n_channels=25,
    sequence_length=200,
    inference_n_units=32,
    inference_normalization="layer",
    model_n_units=32,
    model_normalization="layer",
    des_n_units=16,
    des_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.001,
    n_epochs=300,
)

# Simulate data
print("Simulating data")
sim = simulation.HSMM_MVN(
    n_samples=25600,
    n_channels=config.n_channels,
    n_modes=config.n_modes,
    means="zero",
    covariances="random",
    observation_error=0.0,
    gamma_shape=10,
    gamma_scale=5,
    random_seed=123,
)

sim.standardize()
input_data = data.Data(sim.time_series)

# Plot the transition probability matrix for mode switching in the HSMM
plotting.plot_matrices(
    sim.off_diagonal_trans_prob, filename="figures_longrange/sim_trans_prob.png"
)

# Create tensorflow datasets for training and model evaluation
training_dataset = input_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = input_data.dataset(
    config.sequence_length, config.batch_size, shuffle=False
)

# Build model
model = Model(config)

print("Training model")
history = model.train(training_dataset)


# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_stc = sim.mode_time_course
orders = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, orders[1]]
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

plotting.plot_alpha(
    sim_stc,
    inf_stc,
    y_labels=["Ground Truth", "Sage"],
    filename="figures_longrange/compare.png",
)

plotting.plot_mode_lifetimes(
    sim_stc,
    x_label="Lifetime",
    y_label="Occurrence",
    filename="figures_longrange/sim_lt.png",
)
plotting.plot_mode_lifetimes(
    inf_stc,
    x_label="Lifetime",
    y_label="Occurrence",
    filename="figures_longrange/inf_lt.png",
)

# Ground truth vs inferred covariances
sim_cov = sim.covariances
inf_cov = model.get_covariances()[orders[1]]

plotting.plot_matrices(sim_cov, filename="figures_longrange/sim_cov.png")
plotting.plot_matrices(inf_cov, filename="figures_longrange/inf_cov.png")


# Predict the prior alpha on infer alpha input
gen_alp = model.gen_alpha(inf_alp)
gen_stc = modes.time_courses(gen_alp)

plotting.plot_mode_lifetimes(
    gen_stc,
    x_label="Lifetime",
    y_label="Occurrence",
    filename="figures_longrange/gen_lt.png",
)

plotting.plot_alpha(
    gen_stc,
    y_labels=["Sampled_alpha"],
    filename="figures_longrange/generated_alpha.png",
)
