"""Example script for training DIVE on HMM simulated data.

- Should achieve a dice score of ~0.99.
"""

print("Importing packages")

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.dive import Config, Model
from osl_dynamics.utils import plotting, set_random_seed

set_random_seed(0)

# Create directory to hold plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=20,
    n_sessions=100,
    embeddings_dim=5,
    spatial_embeddings_dim=2,
    sequence_length=200,
    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    dev_n_layers=5,
    dev_n_units=32,
    dev_activation="tanh",
    dev_normalization="layer",
    dev_regularizer="l1",
    dev_regularizer_factor=0.01,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
    batch_size=32,
    learning_rate=0.005,
    lr_decay=0.1,
    n_epochs=40,
)

# Simulate data
print("Simulating data")
sim = simulation.MSess_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    session_means="zero",
    session_covariances="random",
    n_states=config.n_modes,
    n_channels=config.n_channels,
    n_covariances_act=5,
    n_sessions=config.n_sessions,
    embeddings_dim=config.embeddings_dim,
    spatial_embeddings_dim=config.spatial_embeddings_dim,
    embeddings_scale=0.005,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
)
sim.standardize()
training_data = data.Data(sim.time_series)
training_data.add_session_labels(
    "session_id", np.arange(config.n_sessions), "categorical"
)

# Build model
model = Model(config)
model.summary()

# Set regularizers
model.set_regularizers(training_data)

# Set initializer for session-specific deviation parameters
model.set_dev_parameters_initializer(training_data)

# Model initialization
model.random_subset_initialization(training_data, n_init=5, n_epochs=3, take=1)

# Full training
print("Training model")
history = model.fit(training_data)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(training_data, concatenate=True)
inf_stc = modes.argmax_time_courses(inf_alp)
sim_stc = np.concatenate(sim.mode_time_course)

sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (DIVE):", modes.fractional_occupancies(inf_stc))

# Plot the simulated and inferred embeddings with group labels
sim_embeddings = sim.embeddings
inf_embeddings = model.get_summed_embeddings()
lda_inf_embeddings = LinearDiscriminantAnalysis(n_components=2).fit_transform(
    inf_embeddings, sim.assigned_groups
)
group_masks = [sim.assigned_groups == i for i in range(sim.n_groups)]

fig, axes = plotting.create_figure(1, 2, figsize=(10, 5))
plotting.plot_scatter(
    [sim_embeddings[group_mask, 0] for group_mask in group_masks],
    [sim_embeddings[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(config.n_sessions)])[group_mask]
        for group_mask in group_masks
    ],
    ax=axes[0],
)

# Perform PCA on the embeddings to visualise the embeddings
plotting.plot_scatter(
    [lda_inf_embeddings[group_mask, 0] for group_mask in group_masks],
    [lda_inf_embeddings[group_mask, 1] for group_mask in group_masks],
    x_label="dim1",
    y_label="dim2",
    annotate=[
        np.array([str(i) for i in range(config.n_sessions)])[group_mask]
        for group_mask in group_masks
    ],
    ax=axes[1],
)
axes[0].set_title("Simulation")
axes[1].set_title("Inferred")
plotting.save(fig, filename="figures/embeddings.png")
