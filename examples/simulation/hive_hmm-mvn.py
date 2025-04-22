"""Example script for training HIVE on HMM simulated data.

- Should achieve a dice score close to 1.
"""

print("Importing packages")

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.hive import Config, Model
from osl_dynamics.utils import plotting, set_random_seed


set_random_seed(0)

# Directory for plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=5,
    n_channels=20,
    sequence_length=100,
    n_sessions=100,
    embeddings_dim=5,
    spatial_embeddings_dim=2,
    dev_n_layers=5,
    dev_n_units=32,
    dev_activation="tanh",
    dev_normalization="layer",
    dev_regularizer="l1",
    dev_regularizer_factor=0.01,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.001,
    lr_decay=0.1,
    n_epochs=40,
    learn_trans_prob=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
)

# Simulate data
sim = simulation.MSess_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    session_means="zero",
    session_covariances="random",
    n_states=config.n_states,
    n_channels=config.n_channels,
    n_covariances_act=5,
    n_sessions=config.n_sessions,
    embeddings_dim=config.embeddings_dim,
    spatial_embeddings_dim=config.spatial_embeddings_dim,
    embeddings_scale=0.002,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
)
sim.standardize()
sim_stc = np.concatenate(sim.mode_time_course)

# Create training dataset
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
model.random_state_time_course_initialization(
    training_data, n_epochs=3, n_init=10, take=1
)

# Full model training
print("Training model")
history = model.fit(training_data)

# Loss
plotting.plot_line(
    [range(1, len(history["loss"]) + 1)],
    [history["loss"]],
    x_label="Epoch",
    y_label="Loss",
    filename="figures/loss.png",
)

# Get inferred parameters
inf_stc = model.get_alpha(training_data, concatenate=True)
inf_tp = model.get_trans_prob()

# Re-order with respect to the simulation
_, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]
inf_tp = inf_tp[np.ix_(order, order)]


plotting.plot_alpha(
    sim_stc,
    inf_stc,
    n_samples=2000,
    y_labels=["Ground Truth", "Inferred"],
    filename="figures/stc.png",
)

plotting.plot_matrices([inf_tp], filename="figures/trans_prob.png")

# Compare the inferred mode time course to the ground truth
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (Inferred):", modes.fractional_occupancies(inf_stc))

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

plotting.plot_scatter(
    [lda_inf_embeddings[group_mask, 0] for group_mask in group_masks],
    [lda_inf_embeddings[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(config.n_sessions)])[group_mask]
        for group_mask in group_masks
    ],
    ax=axes[1],
)
axes[0].set_title("Simulation")
axes[1].set_title("Inferred")
plotting.save(fig, filename="figures/embeddings.png")
