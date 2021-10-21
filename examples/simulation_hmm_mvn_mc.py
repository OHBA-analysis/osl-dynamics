"""
Example script for running inference on simulated HMM-MVN data.
Multiple scales version for simulation_hmm_mvn.py
"""
print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models import Config, Model
import matplotlib.pyplot as plt
# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2

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
    learn_means = False,
    learn_vars = True,
    learn_fcs = True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Load mode transition probability matrix and covariances of each mode
trans_prob = np.load(files.example.path / "hmm_trans_prob.npy")
cov = np.load(files.example.path / "hmm_cov.npy")

print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=n_samples,
    trans_prob=trans_prob,
    means="zero",
    covariances=cov,
    observation_error=observation_error,
    random_seed=123,
    multiple_scale=True,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

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

history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
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

print("Dice coefficient for mean:", metrics.dice_coefficient(sim_stc_alpha, inf_stc_alpha))
print("Dice coefficient for variance:", metrics.dice_coefficient(sim_stc_beta, inf_stc_beta))
print("Dice coefficient for fc:", metrics.dice_coefficient(sim_stc_gamma, inf_stc_gamma))

# Fractional occupancies
print("Fractional occupancies mean (Simulation):", modes.fractional_occupancies(sim_stc_alpha))
print("Fractional occupancies mean (DyNeMo):      ", modes.fractional_occupancies(inf_stc_alpha))


print("Fractional occupancies variance (Simulation):", modes.fractional_occupancies(sim_stc_beta))
print("Fractional occupancies variance (DyNeMo):      ", modes.fractional_occupancies(inf_stc_beta))


print("Fractional occupancies fc (Simulation):", modes.fractional_occupancies(sim_stc_gamma))
print("Fractional occupancies fc (DyNeMo):      ", modes.fractional_occupancies(inf_stc_gamma))

# plotting the loss over epochs
history_dict = history.history

loss_history = history_dict["loss"]
kl_loss_history = history_dict["kl_loss"]
ll_loss_history = history_dict["ll_loss"]

plt.figure()
plt.plot(loss_history, label="loss")
plt.plot(ll_loss_history, label="ll_loss")
plt.legend()
plt.savefig("figures/total_loss_history.png")


# Delete the temporary folder holding the data
meg_data.delete_dir()
