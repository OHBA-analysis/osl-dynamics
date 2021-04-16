"""Example script for running inference on simulated HMM-MAR data.

- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import config, RIMARO

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 51200

inference_network = config.RNN(
    rnn="lstm",
    n_layers=1,
    n_units=64,
    dropout_rate=0.0,
    normalization="layer",
)

model_network = config.RNN(
    rnn="lstm",
    n_layers=1,
    n_units=64,
    dropout_rate=0.0,
    normalization="layer",
)

alpha = config.Alpha(
    theta_normalization=None,
    xform="softmax",
    learn_temperature=False,
    initial_temperature=2.0,
)

observation_model = config.ObservationModel(
    model="multivariate_autoregressive",
    n_lags=2,
)

kl_annealing = config.KLAnnealing(
    do=True,
    curve="tanh",
    sharpness=15,
    n_epochs=600,
)

training = config.Training(
    batch_size=64,
    learning_rate=0.001,
    n_epochs=800,
)

# MAR parameters
A11 = [[0.9, 0], [0.16, 0.8]]
A12 = [[-0.5, 0], [-0.2, -0.5]]

A21 = [[0.6, 0.1], [0.1, -0.2]]
A22 = [[0.4, 0], [-0.1, 0.1]]

A31 = [[1, -0.15], [0, 0.7]]
A32 = [[-0.3, -0.2], [0.5, 0.5]]

C1 = [1, 1]
C2 = [0.1, 0.1]
C3 = [10, 10]

coeffs = np.array([[A11, A12], [A21, A22], [A31, A32]])
cov = np.array([np.diag(C1), np.diag(C2), np.diag(C3)])

# Simulate data
print("Simulating data")
sim = simulation.HMM_MAR(
    n_samples=n_samples,
    trans_prob="sequence",
    stay_prob=0.95,
    coeffs=coeffs,
    cov=cov,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

dimensions = config.Dimensions(
    n_states=sim.n_states,
    n_channels=meg_data.n_channels,
    sequence_length=200,
)


# Prepare dataset
training_dataset = meg_data.training_dataset(
    dimensions.sequence_length, training.batch_size
)
prediction_dataset = meg_data.prediction_dataset(
    dimensions.sequence_length, training.batch_size
)

# Build model
model = RIMARO(
    dimensions,
    inference_network,
    model_network,
    alpha,
    observation_model,
    kl_annealing,
    training,
)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=training.n_epochs)

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
print("Fractional occupancies (Simulation):", metrics.fractional_occupancies(sim_stc))
print("Fractional occupancies (VRAD):      ", metrics.fractional_occupancies(inf_stc))

# Delete the temporary folder holding the data
meg_data.delete_dir()
