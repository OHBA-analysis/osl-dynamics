"""Example script for running inference using a vector quantized version of DyNeMo
 on simulated HMM-MVN data.

- Achieves a dice of ~0.75.
- There is a large run-to-run variability.
"""

print("Setting up")
import numpy as np
from dynemo import data, simulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600

config = Config(
    n_modes=5,
    n_channels=80,
    sequence_length=100,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization=None,
    n_quantized_vectors=5,
    alpha_xform="gumbel-softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=8,
    learning_rate=0.01,
    n_epochs=100,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=n_samples,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

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
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
)

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_stc = sim.mode_time_course

sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (DyNeMo):      ", modes.fractional_occupancies(inf_stc))

# Load at the inferred quantized vectors
quant_alp = model.get_quantized_alpha()
print("Quantized vectors:")
print(quant_alp)

# Delete temporary directory
meg_data.delete_dir()
