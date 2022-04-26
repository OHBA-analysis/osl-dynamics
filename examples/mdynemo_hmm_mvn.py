"""Example script for running inference on simulated MS-HMM-MVN data.

- Multiple scale version for simulation_hmm_mvn.py
- Should achieve a dice of close to one for alpha and gamma.
"""
print("Setting up")
import numpy as np
from ohba_models import data, simulation
from ohba_models.inference import metrics, modes, tf_ops
from ohba_models.models.mdynemo import Config, Model
from ohba_models.inference import callbacks
from ohba_models.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=40,
    sequence_length=200,
    inference_n_units=128,
    inference_n_layers=2,
    inference_normalization="layer",
    model_n_units=128,
    model_n_layers=2,
    model_normalization="layer",
    theta_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=True,
    learn_stds=True,
    learn_fcs=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=300,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=400,
)

# Simulate data
print("Simulating data")
sim = simulation.MS_HMM_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="random",
    covariances="random",
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Prepare datasets
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)
model.summary()

# Callbacks
dice_callback = callbacks.DiceCoefficientCallback(
    prediction_dataset, sim.mode_time_course, mode_names=["alpha", "gamma"]
)

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
    callbacks=[dice_callback],
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred parameters
inf_means, inf_stds, inf_fcs = model.get_means_stds_fcs()

plotting.plot_matrices(sim.fcs, filename="sim_fcs.png")
plotting.plot_matrices(inf_fcs, filename="inf_fcs.png")

# Inferred mode mixing factors
inf_alpha, inf_gamma = model.get_mode_time_courses(prediction_dataset)

inf_alpha = modes.time_courses(inf_alpha)
inf_gamma = modes.time_courses(inf_gamma)

# Simulated mode mixing factors
sim_alpha, sim_gamma = sim.mode_time_course

# Match the inferred and simulated mixing factors
sim_alpha, inf_alpha = modes.match_modes(sim_alpha, inf_alpha)
sim_gamma, inf_gamma = modes.match_modes(sim_gamma, inf_gamma)

# Dice coefficients
dice_alpha = metrics.dice_coefficient(sim_alpha, inf_alpha)
dice_gamma = metrics.dice_coefficient(sim_gamma, inf_gamma)

print("Dice coefficient for mean:", dice_alpha)
print("Dice coefficient for fc:", dice_gamma)

# Fractional occupancies
fo_sim_alpha = modes.fractional_occupancies(sim_alpha)
fo_sim_gamma = modes.fractional_occupancies(sim_gamma)

fo_inf_alpha = modes.fractional_occupancies(inf_alpha)
fo_inf_gamma = modes.fractional_occupancies(inf_gamma)

print("Fractional occupancies mean (Simulation):", fo_sim_alpha)
print("Fractional occupancies mean (DyNeMo):    ", fo_inf_alpha)

print("Fractional occupancies fc (Simulation):", fo_sim_gamma)
print("Fractional occupancies fc (DyNeMo):    ", fo_inf_gamma)

# Delete the temporary folder holding the data
meg_data.delete_dir()
