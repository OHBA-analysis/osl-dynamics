print("Setting up")
import pickle
import numpy as np
from vrad import data
from vrad.inference import tf_ops
from vrad.simulation import HMM_MAR
from vrad.models import MARO

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 51200
sequence_length = 100
batch_size = 16
n_epochs = 200
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

n_states = sim.n_states
n_lags = sim.order
n_channels = meg_data.n_channels

# Create dataset
training_dataset = meg_data.training_dataset(
    sequence_length,
    batch_size,
    alpha=[sim.state_time_course],
)

# Build model
model = MARO(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    n_lags=n_lags,
    learning_rate=learning_rate,
)
model.summary()

# Train the observation model
history = model.fit(training_dataset, epochs=n_epochs)

inf_coeffs, inf_cov = model.get_params()

print("Ground truth:")
print(np.squeeze(coeffs))
print()

print("Inferred:")
print(np.squeeze(inf_coeffs))

# Delete the temporary folder holding the data
meg_data.delete_dir()
