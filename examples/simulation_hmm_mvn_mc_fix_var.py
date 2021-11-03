"""
Example script for running inference on simulated HMM-MVN data.
Multiple scales version for simulation_hmm_mvn.py
But now we vary the mean, fix the variance.
"""
print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models import Config, Model
import matplotlib.pyplot as plt
from dynemo.inference import callbacks

# GPU settings
tf_ops.gpu_growth()


# Load mode transition probability matrix and covariances of each mode
trans_prob = np.load(files.example.path / "hmm_trans_prob.npy")
cov = np.load(files.example.path / "hmm_cov.npy")

# cov.shape = (n_modes, n_channels, n_channels)


# Settings
n_samples = 25600
observation_error = 0.2

print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=n_samples,
    trans_prob=trans_prob,
    means="random",
    n_modes=cov.shape[0],
    n_channels=cov.shape[-1],
    covariances="random",
    observation_error=observation_error,
    random_seed=123,
    multiple_scale=True,
    fix_variance=True,
)
sim.standardize()
meg_data = data.Data(sim.time_series)



config = Config(
    multiple_scale=True,
    n_modes=5,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
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
    learning_rate=0.01,
    n_epochs=200,
    fix_variance=True,
)


config.n_channels = meg_data.n_channels

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

print("Training model")

dice_callback = callbacks.DiceCoefficientCallbackMultipleScale(
    prediction_dataset, sim.mode_time_course
)

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

# plotting the loss over epochs
history_dict = history.history

loss_history = history_dict["loss"]
kl_loss_history = history_dict["kl_loss"]
ll_loss_history = history_dict["ll_loss"]
dice_alpha_history = history_dict["dice_alpha"]
dice_beta_history = history_dict["dice_beta"]
dice_gamma_history = history_dict["dice_gamma"]

plt.figure()
plt.plot(loss_history, label="loss")
plt.plot(ll_loss_history, label="ll_loss")
plt.title("total loss and ll loss against epoch")
plt.legend()
plt.savefig("figures/total_loss_history.png")

plt.figure()
plt.plot(dice_alpha_history, label="dice_alpha")
plt.plot(dice_gamma_history, label="dice_gamma")
plt.title(
    "dice score of inferred mean (alpha) and fc (gamma) time courses against epoch"
)
plt.legend()
plt.savefig("figures/dice_history.png")

# Delete the temporary folder holding the data
meg_data.delete_dir()
