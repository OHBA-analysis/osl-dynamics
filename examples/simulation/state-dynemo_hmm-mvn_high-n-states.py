"""Example script for training State-DyNeMo on HMM-MVN simulated data.

"""

print("Setting up")
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.state_dynemo import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=14,
    n_channels=80,
    sequence_length=400,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_states=config.n_states,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim.standardize()
training_data = data.Data(sim.time_series)

# Build model
model = Model(config)
model.summary()

# Initialization
model.random_state_time_course_initialization(training_data, n_epochs=20, n_init=5)

print("Training model")
history = model.fit(training_data)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(training_data)
inf_stc = modes.argmax_time_courses(inf_alp)
sim_stc = sim.state_time_course

sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (State-DyNeMo):", modes.fractional_occupancies(inf_stc))
