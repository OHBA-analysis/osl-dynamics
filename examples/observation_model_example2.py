print("Setting up")
import pickle
import numpy as np
from vrad import data
from vrad.inference import tf_ops
from vrad.simulation import HMM_MAR
from vrad.models import MARO

# GPU settings
tf_ops.gpu_growth()
multi_gpu = True

# Settings
n_samples = 25600
sequence_length = 100
batch_size = 16
n_epochs = 20
learning_rate = 0.01

# MAR parameters
A1 = [[0.9, 0], [0.16, 0.8]]
A2 = [[-0.5, 0], [-0.2, -0.5]]
C = [0.2, 0.5]

coeffs = np.array([[A1, A2]])
cov = np.array([np.diag(C)])

# Simulate data
print("Simulating data")
sim = HMM_MAR(
    n_samples=n_samples,
    trans_prob=None,
    coeffs=coeffs,
    cov=cov,
    random_seed=123,
)
# sim.standardize()
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
    multi_gpu=multi_gpu,
)
model.summary()

# Train the observation model
history = model.fit(training_dataset, epochs=n_epochs)

inf_coeffs, inf_cov = model.get_params()

print("Ground truth:")
print(np.squeeze(coeffs))
print(np.squeeze(cov))
print()

print("Inferred:")
print(np.squeeze(inf_coeffs))
print(np.squeeze(inf_cov))

# Delete the temporary folder holding the data
meg_data.delete_dir()
