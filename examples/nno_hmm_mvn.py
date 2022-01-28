"""Example script for fitting a multivariate normal observation model to data.

"""

print("Setting up")
from dynemo import data, simulation
from dynemo.inference import tf_ops
from dynemo.models.nno import Config, Model
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=5,
    n_channels=11,
    sequence_length=200,
    mlp_n_layers=2,
    mlp_n_units=128,
    mlp_normalization="batch",
    mlp_activation="selu",
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
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
    alpha=[sim.mode_time_course],
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)
