"""Example script for running HMM inference on simulated HMM-MVN data.

"""

import os

import numpy as np
from matplotlib import pyplot as plt
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.sehmm import Config, Model
from osl_dynamics.utils import plotting

# Directory for plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.select_gpu(0)
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=5,
    n_channels=20,
    sequence_length=200,
    n_subjects=100,
    subject_embeddings_dim=2,
    mode_embeddings_dim=2,
    dev_n_layers=5,
    dev_n_units=32,
    dev_activation="tanh",
    dev_normalization="layer",
    dev_regularizer="l1",
    dev_regularizer_factor=10,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=1e-3,
    n_epochs=30,
    learn_trans_prob=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=15,
)

# Simulate data
sim = simulation.MSubj_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    subject_means="zero",
    subject_covariances="random",
    n_states=config.n_states,
    n_channels=config.n_channels,
    n_covariances_act=2,
    n_subjects=config.n_subjects,
    n_subject_embedding_dim=config.subject_embeddings_dim,
    n_mode_embedding_dim=config.mode_embeddings_dim,
    subject_embedding_scale=0.002,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
    random_seed=1234,
)
sim.standardize()
sim_stc = np.concatenate(sim.mode_time_course)

# Create training dataset
training_data = data.Data([mtc for mtc in sim.time_series])

# Build model
model = Model(config)
model.summary()

# Set regularizers
model.set_regularizers(training_data)

model.random_subset_initialization(training_data, n_epochs=3, n_init=3, take=0.3)
# Train model
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

plotting.plot_matrices(inf_tp[np.newaxis, ...], filename="figures/trans_prob.png")

# Compare the inferred mode time course to the ground truth
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (Inferred):", modes.fractional_occupancies(inf_stc))


sim_subject_embeddings = sim.subject_embeddings
subject_embeddings = model.get_subject_embeddings()
group_masks = [sim.assigned_groups == i for i in range(sim.n_groups)]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plotting.plot_scatter(
    [sim_subject_embeddings[group_mask, 0] for group_mask in group_masks],
    [sim_subject_embeddings[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(config.n_subjects)])[group_mask]
        for group_mask in group_masks
    ],
    ax=axes[0],
)

plotting.plot_scatter(
    [subject_embeddings[group_mask, 0] for group_mask in group_masks],
    [subject_embeddings[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(config.n_subjects)])[group_mask]
        for group_mask in group_masks
    ],
    ax=axes[1],
)
axes[0].set_title("Simulation")
axes[1].set_title("Inferred")
fig.savefig("figures/subject_embeddings.png")
plt.close()

training_data.delete_dir()
