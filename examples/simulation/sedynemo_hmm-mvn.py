"""Example script for running inference on simulated multi-subject HMM-MVN data.

-Should achieve a dice score of 0.99.
"""

print("Setting up")
import numpy as np
from sklearn.decomposition import PCA
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.utils import plotting
from osl_dynamics.models.sedynemo import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=20,
    n_subjects=20,
    subject_embedding_dim=5,
    mode_embedding_dim=2,
    sequence_length=200,
    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=64,
    learning_rate=0.005,
    n_epochs=200,
    multi_gpu=False,
)

# Simulate data
print("Simulating data")
print("Number of GPUs in use:", config.strategy.num_replicas_in_sync)

sim = simulation.MSubj_HMM_MVN(
    n_samples=12800,
    n_subjects=config.n_subjects,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim.standardize()
training_data = data.Data([mtc for mtc in sim.time_series])

# Prepare dataset
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size * config.strategy.num_replicas_in_sync,
    shuffle=True,
    subj_id=True,
)
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size * config.strategy.num_replicas_in_sync,
    shuffle=False,
    subj_id=True,
)

# Bayesian about subject deviations?
config.dev_bayesian = True
config.learn_dev_mod_sigma = True
config.initial_dev_mod_sigma = np.sqrt(0.5 / 1e-3)

# Build model
model = Model(config)
model.summary()

# Set scaling factor for devation kl loss
model.set_bayesian_kl_scaling(training_dataset)

# Set regularizers
model.set_regularizers(training_dataset)

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_stc = np.concatenate(sim.mode_time_course)

sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (DyNeMo):      ", modes.fractional_occupancies(inf_stc))

# Get the subject embeddings
subject_embeddings = model.get_subject_embeddings()

# Perform PCA on the subject embeddings to visualise the embeddings
pca = PCA(n_components=2)
pca_subject_embeddings = pca.fit_transform(subject_embeddings)
print("explained variances ratio:", pca.explained_variance_ratio_)
plotting.plot_scatter(
    [pca_subject_embeddings[:, 0]],
    [pca_subject_embeddings[:, 1]],
    x_label="PC1",
    y_label="PC2",
    annotate=[[str(i) for i in range(config.n_subjects)]],
    filename="subject_embeddings.png",
)

training_data.delete_dir()
