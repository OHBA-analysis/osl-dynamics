"""Example script for running DyNeMo on simulated HMM-MVN data with
a large number of states.

"""

print("Setting up")
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.dynemo import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=12,
    n_channels=80,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
sim.standardize()
training_data = data.Data(sim.time_series)

# Build model
model = Model(config)
model.summary()

# Initialization
init_history = model.random_subset_initialization(
    training_data,
    n_epochs=20,
    n_init=5,
    take=0.25,
)

print("Training model")
history = model.fit(training_data)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and state time course
inf_alp = model.get_alpha(training_data)
inf_stc = modes.argmax_time_courses(inf_alp)
sim_stc = sim.mode_time_course

# Inferred covariances
inf_cov = model.get_covariances()
sim_cov = sim.covariances

# Reorder inferred modes to match the simulation
_, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]
inf_cov = inf_cov[order]

# Metrics
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (DyNeMo):", modes.fractional_occupancies(inf_stc))
