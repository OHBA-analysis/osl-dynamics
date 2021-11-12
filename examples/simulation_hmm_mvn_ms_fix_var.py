"""Example script for running inference on simulated HMM-MVN data.

- Multiple scale version for simulation_hmm_mvn.py
- We vary the mean, fix the variance.
"""
print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models import Config, Model
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
sim = simulation.HMM_MVN(
    n_samples=n_samples,
    trans_prob=trans_prob,
    means="random",
    n_modes=n_modes,
    n_channels=n_channels,
    covariances="random",
    observation_error=observation_error,
    random_seed=123,
    multiple_scale=True,
    fix_variance=True,
    uni_variance=True,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Hyperparameters
config = Config(
    multiple_scale=True,
    n_modes=n_modes,
    n_channels=n_channels,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=128,
    inference_normalization="layer",
    inference_dropout_rate=0.4,
    model_rnn="lstm",
    model_n_units=128,
    model_normalization="layer",
    model_dropout_rate=0.4,
    theta_normalization="layer",
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=True,
    learn_vars=True,
    learn_fcs=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=150,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=300,
    fix_variance=True,
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
print("Building Model")
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

# Inferred mode mixing factors and mode time course
inf_alpha, inf_beta, inf_gamma = model.get_alpha(prediction_dataset)

inf_stc_alpha = modes.time_courses(inf_alpha)
inf_stc_beta = modes.time_courses(inf_beta)
inf_stc_gamma = modes.time_courses(inf_gamma)

sim_stc = sim.mode_time_course
sim_stc_alpha = sim_stc[:, :, 0]
sim_stc_beta = sim_stc[:, :, 1]
sim_stc_gamma = sim_stc[:, :, 2]

sim_stc_alpha, inf_stc_alpha = modes.match_modes(sim_stc_alpha, inf_stc_alpha)
sim_stc_beta, inf_stc_beta = modes.match_modes(sim_stc_beta, inf_stc_beta)
sim_stc_gamma, inf_stc_gamma = modes.match_modes(sim_stc_gamma, inf_stc_gamma)

print(
    "Dice coefficient for mean:", metrics.dice_coefficient(sim_stc_alpha, inf_stc_alpha)
)
print(
    "Dice coefficient for variance:",
    metrics.dice_coefficient(sim_stc_beta, inf_stc_beta),
)
print(
    "Dice coefficient for fc:", metrics.dice_coefficient(sim_stc_gamma, inf_stc_gamma)
)

# Fractional occupancies
print(
    "Fractional occupancies mean (Simulation):",
    modes.fractional_occupancies(sim_stc_alpha),
)
print(
    "Fractional occupancies mean (DyNeMo):      ",
    modes.fractional_occupancies(inf_stc_alpha),
)


print(
    "Fractional occupancies variance (Simulation):",
    modes.fractional_occupancies(sim_stc_beta),
)
print(
    "Fractional occupancies variance (DyNeMo):      ",
    modes.fractional_occupancies(inf_stc_beta),
)


print(
    "Fractional occupancies fc (Simulation):",
    modes.fractional_occupancies(sim_stc_gamma),
)
print(
    "Fractional occupancies fc (DyNeMo):      ",
    modes.fractional_occupancies(inf_stc_gamma),
)

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
    + "lr={config.learning_rate}, n_units={config.inference_n_units}, "
    + "n_layers={config.inference_n_layers}, drop_out={config.inference_dropout_rate},\n"
    + "n_epochs={config.n_epochs}, annealing_epochs={config.n_kl_annealing_epochs}",
    filename="figures/loss.png",
)

plotting.plot_line(
    [range(config.n_epochs), range(config.n_epochs)],
    [dice_alpha, dice_gamma],
    labels=["dice_alpha", "dice_gamma"],
    title=f"dice scores against epoch\n"
    + "lr={config.learning_rate}, n_units={config.inference_n_units}, "
    + "n_layers={config.inference_n_layers}, drop_out={config.inference_dropout_rate},\n"
    + "n_epochs={config.n_epochs}, annealing_epochs={config.n_kl_annealing_epochs}",
    filename="figures/dice.png",
)

# Delete the temporary folder holding the data
meg_data.delete_dir()
