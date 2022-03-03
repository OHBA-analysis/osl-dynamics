"""Example script for running inference on simulated HMM-MVN data.
Subject embedding version of dyneme_hmm_mvn.py
Should achieve a dice score of 0.99
"""

print("Setting up")
import numpy as np
from ohba_models import data, simulation
from ohba_models.inference import metrics, modes, tf_ops
from ohba_models.models.se_dynemo import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=11,
    n_subjects=10,
    embedding_dim=3,
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
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Simulate data
print("Simulating data")
sim = simulation.MultiSubject_HMM_MVN(
    n_samples = 25600,
    n_subjects=10,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim.standardize()
meg_data = data.Data([mtc for mtc in sim.time_series])

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
    subj_id=True,
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
    subj_id=True,
)

# Build model
model = Model(config)
model.summary()

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

meg_data.delete_dir()