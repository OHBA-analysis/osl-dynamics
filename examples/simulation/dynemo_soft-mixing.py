"""Example script for demonstrating DyNeMo's ability to infer a soft mixture of modes.

"""

print("Setting up")
import os
import sys
import numpy as np
from tqdm.auto import trange

from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.utils import plotting

# Create directory to hold plots
save_dir = sys.argv[1]
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=6,
    n_channels=80,
    sequence_length=200,
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
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    #relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    #amplitudes=[6, 5, 4, 3, 2, 1],
    #frequencies=[1, 2, 3, 4, 6, 8],
    relative_activation=[1, 1, 1, 1, 1, 1],
    amplitudes=[1, 1, 1, 1, 1, 1],
    frequencies=[1.2, 2.2, 3.2, 4.2, 6.2, 8.2],
    sampling_frequency=250,
    means="zero",
    covariances="random",
)
sim_alp = sim.mode_time_course
training_data = data.Data(sim.time_series)

np.save(f'{save_dir}/mode_time_course.npy',sim_alp)
np.save(f'{save_dir}/time_series.npy',sim.time_series)
np.save(f'{save_dir}/ground_truth_covs.npy',sim.covariances)

# Plot ground truth logits
plotting.plot_separate_time_series(
    sim.logits, n_samples=2000, filename=f"{save_dir}/sim_logits.png"
)

# Build model
model = Model(config)
model.summary()
model.set_covariances(sim.covariances)

print("Training model")
history = model.fit(
    training_data,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath=f"{save_dir}/tmp/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")
model.save(f'{save_dir}/model/')
with open(f"{save_dir}/free_energy.txt", "w") as f:
    f.write(f"{free_energy:.6f}")


# Inferred alpha and mode time course
inf_alp = model.get_alpha(training_data)
orders = modes.match_modes(sim_alp, inf_alp, return_order=True)

np.save(f'{save_dir}/alp.npy',inf_alp)
np.save(f'{save_dir}/covs.npy',model.get_covariances())


inf_alp = inf_alp[:, orders[1]]

# Compare the inferred mode time course to the ground truth
plotting.plot_alpha(
    sim_alp,
    n_samples=2000,
    title="Ground Truth",
    y_labels=r"$\alpha_{jt}$",
    filename=f"{save_dir}/sim_alp.png",
)
plotting.plot_alpha(
    inf_alp,
    n_samples=2000,
    title="DyNeMo",
    y_labels=r"$\alpha_{jt}$",
    filename=f"{save_dir}/inf_alp.png",
)

# Correlation between mode time courses
corr = metrics.alpha_correlation(inf_alp, sim_alp)
print("Correlation (DyNeMo vs Simulation):", corr)

# Reconstruction of the time-varying covariance
sim_cov = sim.covariances
inf_cov = model.get_covariances()[orders[1]]

sim_alp = sim_alp[:2000]
inf_alp = inf_alp[:2000]
sim_tvcov = np.sum(
    sim_alp[:, :, np.newaxis, np.newaxis] * sim_cov[np.newaxis, :, :, :], axis=1
)
inf_tvcov = np.sum(
    inf_alp[:, :, np.newaxis, np.newaxis] * inf_cov[np.newaxis, :, :, :], axis=1
)

# Calculate the Riemannian distance between the ground truth and inferred covariance
print("Calculating riemannian distances")
rd = np.empty(2000)
for i in trange(2000):
    rd[i] = metrics.riemannian_distance(sim_tvcov[i], inf_tvcov[i])

plotting.plot_line(
    [range(2000)],
    [rd],
    labels=["DyNeMo"],
    x_label="Sample",
    y_label="$d$",
    fig_kwargs={"figsize": (15, 1.5)},
    filename=f"{save_dir}/rd.png",
)

# Delete temporary directory
training_data.delete_dir()
