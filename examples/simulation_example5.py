"""Example script for running inference on simulated HMM-MAR data.

- Should achieve a dice coefficient of ~...
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from vrad import data
from vrad.inference import metrics, states, tf_ops
from vrad.models import RIMARO
from vrad.simulation import HMM_MAR

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
n_lags = 2

n_states = 3
sequence_length = 200
batch_size = 16

do_annealing = True
annealing_sharpness = 10

n_epochs = 100
n_epochs_annealing = 50

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = None

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 64

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_covariances = True

alpha_xform = "softmax"
alpha_temperature = 0.2
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

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
sim = HMM_MAR(
    n_samples=n_samples,
    trans_prob="sequence",
    stay_prob=0.95,
    coeffs=coeffs,
    cov=cov,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim)
n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Build model
model = RIMARO(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    n_lags=n_lags,
    rnn_type=rnn_type,
    rnn_normalization=rnn_normalization,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    theta_normalization=theta_normalization,
    alpha_xform=alpha_xform,
    alpha_temperature=alpha_temperature,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    learning_rate=learning_rate,
    initial_cov=np.array([C1, C2, C3]),
    learn_cov=False,
)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=n_epochs)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
inf_alpha = model.predict_states(prediction_dataset)[0]
inf_stc = states.time_courses(inf_alpha)
sim_stc = sim.state_time_course

sim_stc, inf_stc = states.match_states(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", metrics.fractional_occupancies(sim_stc))
print("Fractional occupancies (VRAD):      ", metrics.fractional_occupancies(inf_stc))

# Delete the temporary folder holding the data
meg_data.delete_dir()
