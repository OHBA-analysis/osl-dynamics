"""
Example script for stability analysis on the MC model with variance time course fixed.
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

# Settings
n_samples = 25600
observation_error = 0.2

# Simulate data with means and covariances random.
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

# Set up the model
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
    n_kl_annealing_epochs=100,
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

# Set the number of runs for training the model.
N_runs = 10

# Set up a dictionary for storing results for each run.
stability_dict = {
    "free_energy": [],
    "dice_alpha": [],
    "dice_gamma": [],
}

for run in range(N_runs):
    print(f"Starting run {run + 1} of {N_runs} runs......")
    print("Building Model")
    model = Model(config)

    print("Training model")
    history = model.fit(
        training_dataset,
        epochs=config.n_epochs,
        verbose=0,
    )

    # Free energy = Log Likelihood - KL Divergence
    free_energy = model.free_energy(prediction_dataset)

    # Inferred mode mixing factors and mode time course
    inf_alpha, inf_beta, inf_gamma = model.get_alpha(prediction_dataset)

    inf_stc_alpha = modes.time_courses(inf_alpha)
    inf_stc_gamma = modes.time_courses(inf_gamma)

    sim_stc = sim.mode_time_course
    sim_stc_alpha = sim_stc[:, :, 0]
    sim_stc_gamma = sim_stc[:, :, 2]

    sim_stc_alpha, inf_stc_alpha = modes.match_modes(sim_stc_alpha, inf_stc_alpha)
    sim_stc_gamma, inf_stc_gamma = modes.match_modes(sim_stc_gamma, inf_stc_gamma)

    # load in the results to the dictionary
    stability_dict["free_energy"].append(free_energy)
    stability_dict["dice_alpha"].append(
        metrics.dice_coefficient(sim_stc_alpha, inf_stc_alpha)
    )
    stability_dict["dice_gamma"].append(
        metrics.dice_coefficient(sim_stc_gamma, inf_stc_gamma)
    )


# Statistics for the results

print("Total number of runs: ", N_runs)

# Free energy
free_energy_list = stability_dict["free_energy"]
print(
    "Free energy: mean/median (min/max) = ",
    f"{np.mean(free_energy_list)}/{np.median(free_energy_list)} ({np.amin(free_energy_list)}/{np.amax(free_energy_list)})",
)

# Dice_alpha
dice_alpha_list = stability_dict["dice_alpha"]
print(
    "Dice coefficient for mean: mean/median (min/max) = ",
    f"{np.mean(dice_alpha_list)}/{np.median(dice_alpha_list)} ({np.amin(dice_alpha_list)}/{np.amax(dice_alpha_list)})",
)

# Dice_gamma
dice_gamma_list = stability_dict["dice_gamma"]
print(
    "Dice coefficient for fc: mean/median (min/max) = ",
    f"{np.mean(dice_gamma_list)}/{np.median(dice_gamma_list)} ({np.amin(dice_gamma_list)}/{np.amax(dice_gamma_list)})",
)

# Scatter plot of dice scores
plt.figure()
plt.scatter(dice_alpha_list, dice_gamma_list)
plt.xlabel("dice_mean")
plt.ylabel("dice_fc")
plt.title("scatter plot of dice scores")
plt.tight_layout()
plt.savefig("figures/dice_scatter_plot.png")

# plot showing free energy and dice for each run
fig, ax1 = plt.subplots()
x = np.arange(N_runs) + 1

ax1.set_xlabel("runs")
ax1.set_ylabel("dice scores")
ax1.bar(x - 0.1, dice_alpha_list, width=0.2, color="r", align="center", label="dice_alpha")
ax1.bar(x + 0.1, dice_gamma_list, width=0.2, color="b", align="center", label="dice_gamma")

ax2 = ax1.twinx()
ax2.set_ylabel("free energy")
ax2.plot(x, free_energy_list, color="k", label="free energy")
ax2.set_title("plot comparing result from each run")
fig.legend()
fig.tight_layout()
fig.savefig("figures/compare_runs.png")

# Delete the temporary folder holding the data
meg_data.delete_dir()
