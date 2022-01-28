"""Example script for running inference on simulated HMM-MVN data.

- Multiple scale version for simulation_hmm_mvn.py
- We vary the mean, fix the standard deviation.
- Should achieve a dice of close to one for alpha, beta and gamma.
"""
print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models.mrigo import Config, Model
from dynemo.inference import callbacks
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2

# Load mode transition probability matrix and covariances of each mode
trans_prob = np.load(files.example.path / "hmm_trans_prob.npy")
cov = np.load(files.example.path / "hmm_cov.npy")

# Number of modes and channels
n_modes = cov.shape[0]
n_channels = cov.shape[-1]

print("Simulating data")
sim = simulation.MS_HMM_MVN(
    n_samples=n_samples,
    trans_prob=trans_prob,
    means="random",
    n_modes=n_modes,
    n_channels=n_channels,
    covariances="random",
    observation_error=observation_error,
    random_seed=123,
    fix_std=True,
    uni_std=True,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Hyperparameters
config = Config(
    n_modes=n_modes,
    n_channels=n_channels,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=128,
    inference_n_layers=2,
    inference_normalization="layer",
    inference_dropout_rate=0.2,
    model_rnn="lstm",
    model_n_units=128,
    model_n_layers=2,
    model_normalization="layer",
    model_dropout_rate=0.2,
    theta_normalization="layer",
    alpha_xform="softmax",
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
    fix_std=True,
    separate_rnns=False,
)

# Prepare dataset
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
    prediction_dataset, sim.mode_time_course, mode_names=["alpha", "beta", "gamma"]
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

# Infered means, standard deviations and functional connectivities
means, stds, fcs = model.get_means_stds_fcs()

print("means:", means)
print("stds:", stds)
print("fcs:", fcs)

# Inferred mode mixing factors
inf_alpha, inf_beta, inf_gamma = model.get_mode_time_courses(prediction_dataset)

inf_alpha = modes.time_courses(inf_alpha)
inf_beta = modes.time_courses(inf_beta)
inf_gamma = modes.time_courses(inf_gamma)

# Simulated mode mixing factors
sim_alpha, sim_beta, sim_gamma = sim.mode_time_course

# Match the inferred and simulated mixing factors
sim_alpha, inf_alpha = modes.match_modes(sim_alpha, inf_alpha)
sim_beta, inf_beta = modes.match_modes(sim_beta, inf_beta)
sim_gamma, inf_gamma = modes.match_modes(sim_gamma, inf_gamma)

# Dice coefficients
dice_alpha = metrics.dice_coefficient(sim_alpha, inf_alpha)
dice_beta = metrics.dice_coefficient(sim_beta, inf_beta)
dice_gamma = metrics.dice_coefficient(sim_gamma, inf_gamma)

print("Dice coefficient for mean:", dice_alpha)
print("Dice coefficient for std:", dice_beta)
print("Dice coefficient for fc:", dice_gamma)

# Fractional occupancies
fo_sim_alpha = modes.fractional_occupancies(sim_alpha)
fo_sim_beta = modes.fractional_occupancies(sim_beta)
fo_sim_gamma = modes.fractional_occupancies(sim_gamma)

fo_inf_alpha = modes.fractional_occupancies(inf_alpha)
fo_inf_beta = modes.fractional_occupancies(inf_beta)
fo_inf_gamma = modes.fractional_occupancies(inf_gamma)

print("Fractional occupancies mean (Simulation):", fo_sim_alpha)
print("Fractional occupancies mean (DyNeMo):    ", fo_inf_alpha)

print("Fractional occupancies std (Simulation):", fo_sim_beta)
print("Fractional occupancies std (DyNeMo):    ", fo_inf_beta)

print("Fractional occupancies fc (Simulation):", fo_sim_gamma)
print("Fractional occupancies fc (DyNeMo):    ", fo_inf_gamma)

# Plot training history
history = history.history

loss = history["loss"]
kl_loss = history["kl_loss"]
ll_loss = history["ll_loss"]
dice_alpha = history["dice_alpha"]
dice_beta = history["dice_beta"]
dice_gamma = history["dice_gamma"]

plotting.plot_line(
    [range(config.n_epochs), range(config.n_epochs)],
    [loss, ll_loss],
    labels=["loss", "ll_loss"],
    title=f"total loss and ll loss against epoch\n "
    + f"lr={config.learning_rate}, n_units={config.inference_n_units}, "
    + f"n_layers={config.inference_n_layers}, drop_out={config.inference_dropout_rate},\n"
    + f"n_epochs={config.n_epochs}, annealing_epochs={config.n_kl_annealing_epochs}",
    filename="figures/loss.png",
)

plotting.plot_line(
    [range(config.n_epochs), range(config.n_epochs)],
    [dice_alpha, dice_gamma],
    labels=["dice_alpha", "dice_gamma"],
    title=f"dice scores against epoch\n"
    + f"lr={config.learning_rate}, n_units={config.inference_n_units}, "
    + f"n_layers={config.inference_n_layers}, drop_out={config.inference_dropout_rate},\n"
    + f"n_epochs={config.n_epochs}, annealing_epochs={config.n_kl_annealing_epochs}",
    filename="figures/dice.png",
)

# Delete the temporary folder holding the data
meg_data.delete_dir()
