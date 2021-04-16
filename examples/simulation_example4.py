"""Example script for running inference on a hierarchical HMM simulation.

"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import config, RIGO

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2

dimensions = config.Dimensions(
    n_states=5,
    n_channels=80,
    sequence_length=200,
)

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
    initial_temperature=0.25,
    learn_temperature=False,
)

observation_model = config.ObservationModel(
    model="multivariate_normal",
    learn_covariances=True,
    learn_alpha_scaling=False,
    normalize_covariances=False,
)

kl_annealing = config.KLAnnealing(
    do=True,
    curve="tanh",
    sharpness=10,
    n_epochs=50,
)

training = config.Training(
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Load state transition probability matrix and covariances of each state
example_file_directory = Path(__file__).parent / "files"
cov = np.load(example_file_directory / "hmm_cov.npy")

top_level_trans_prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
bottom_level_trans_probs = [
    np.array(
        [
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6],
        ]
    ),
    np.array(
        [
            [0.9, 0.1, 0, 0, 0],
            [0, 0.9, 0.1, 0, 0],
            [0, 0, 0.9, 0.1, 0],
            [0, 0, 0, 0.9, 0.1],
            [0.1, 0, 0, 0, 0.9],
        ]
    ),
    np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    ),
]
# Simulate data
print("Simulating data")
sim = simulation.HierarchicalHMM_MVN(
    n_samples=n_samples,
    top_level_trans_prob=top_level_trans_prob,
    bottom_level_trans_probs=bottom_level_trans_probs,
    means="zero",
    covariances=cov,
    observation_error=observation_error,
    top_level_random_seed=123,
    bottom_level_random_seeds=[124, 126, 127],
    data_random_seed=555,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = meg_data.training_dataset(
    dimensions.sequence_length, training.batch_size
)
prediction_dataset = meg_data.prediction_dataset(
    dimensions.sequence_length, training.batch_size
)

# Build model
model = RIGO(
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
history = model.fit(
    training_dataset,
    epochs=training.n_epochs,
    save_best_after=kl_annealing.n_epochs,
    save_filepath="tmp/weights",
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
print("Fractional occupancies (Simulation):", metrics.fractional_occupancies(sim_stc))
print("Fractional occupancies (VRAD):      ", metrics.fractional_occupancies(inf_stc))

# Delete the temporary folder holding the data
meg_data.delete_dir()
