"""Example script for running inference on a hierarchical HMM simulation.

"""

print("Setting up")
import numpy as np
from vrad import data, files, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2

config = Config(
    n_states=5,
    n_channels=80,
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
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Load state transition probability matrix and covariances of each state
cov = np.load(files.example.path / "hmm_cov.npy")

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

config.n_channel = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
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

# Delete the temporary folder holding the data
meg_data.delete_dir()
