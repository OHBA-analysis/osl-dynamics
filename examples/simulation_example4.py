"""Example script for running inference on a hierarchical HMM simulation.

"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import RIGO

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2

n_states = 5
sequence_length = 200

batch_size = 16
learning_rate = 0.01
n_epochs = 200

do_kl_annealing = True
kl_annealing_sharpness = 10
n_epochs_kl_annealing = 100

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = None

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 64

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

alpha_xform = "softmax"
do_alpha_temperature_annealing = False
alpha_temperature = 0.25

learn_covariances = True
learn_alpha_scaling = False
normalize_covariances = False

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
n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Build model
model = RIGO(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_covariances=learn_covariances,
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
    do_alpha_temperature_annealing=do_alpha_temperature_annealing,
    alpha_temperature=alpha_temperature,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_kl_annealing=do_kl_annealing,
    kl_annealing_sharpness=kl_annealing_sharpness,
    n_epochs_kl_annealing=n_epochs_kl_annealing,
    learning_rate=learning_rate,
)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=n_epochs,
    save_best_after=n_epochs_kl_annealing,
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
