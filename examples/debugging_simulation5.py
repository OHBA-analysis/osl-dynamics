"""Example script for running inference on simulated HMM-MAR data.

- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from vrad import data, simulation
from vrad.inference import callbacks, metrics, states, tf_ops
from vrad.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600

config = Config(
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    observation_model="multivariate_autoregressive",
    n_lags=2,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# MAR parameters
A11 = [[0.9, 0], [0.16, 0.8]]
A12 = [[-0.5, 0], [-0.2, -0.5]]

A21 = [[0.6, 0.1], [0.1, -0.2]]
A22 = [[0.4, 0], [-0.1, 0.1]]

A31 = [[1, -0.15], [0, 0.7]]
A32 = [[-0.3, -0.2], [0.5, 0.5]]

C1 = [1, 0.5]
C2 = [0.1, 0.1]
C3 = [10, 10]

coeffs = np.array([[A11, A12], [A21, A22], [A31, A32]])
covs = np.array([np.diag(C1), np.diag(C2), np.diag(C3)])

# Simulate data
print("Simulating data")
sim = simulation.HMM_MAR(
    n_samples=n_samples,
    trans_prob="sequence",
    stay_prob=0.95,
    coeffs=coeffs,
    covs=covs,
    random_seed=123,
)
meg_data = data.Data(sim.time_series)

config.n_states = sim.n_states
config.n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

config.initial_coeffs = coeffs
config.initial_covs = covs

# Build model
model = Model(config)
model.summary()

# Callbacks
dice_callback = callbacks.DiceCoefficientCallback(
    prediction_dataset, sim.state_time_course
)

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    callbacks=[dice_callback],
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
inf_alpha = model.predict_states(prediction_dataset)
inf_stc = states.time_courses(inf_alpha)
sim_stc = sim.state_time_course

sim_stc, inf_stc = states.match_states(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", states.fractional_occupancies(sim_stc))
print("Fractional occupancies (VRAD):      ", states.fractional_occupancies(inf_stc))

# Inferred parameters
inf_coeffs, inf_covs = model.get_params()

print("Inferred parameters:")
print(coeffs)
print(inf_coeffs)
print()

print(covs)
print(inf_covs)

# Delete the temporary folder holding the data
meg_data.delete_dir()
