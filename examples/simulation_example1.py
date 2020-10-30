"""Example script for running inference on simulated HMM data.

- Should achieve a dice coefficient of ~0.95.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
import numpy as np
from vrad import array_ops, data
from vrad.inference import gmm, metrics, states, tf_ops
from vrad.models import RNNGaussian
from vrad.simulation import HMMSimulation

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25000
observation_error = 0.2

n_states = 5
sequence_length = 50
batch_size = 32

do_annealing = True
annealing_sharpness = 5

n_epochs = 100
n_epochs_annealing = 50

rnn_type = "lstm"
normalization_type = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 16
n_units_model = 24

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Load state transition probability matrix and covariances of each state
trans_prob = np.load("files/hmm_trans_prob.npy")
cov = np.load("files/hmm_cov.npy")

# Simulate data
print("Simulating data")
sim = HMMSimulation(
    n_samples=n_samples,
    n_states=n_states,
    sim_varying_means=learn_means,
    covariances=cov,
    trans_prob=trans_prob,
    observation_error=observation_error,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim)
n_channels = meg_data.n_channels

# Initialsation of means and covariances
initial_means, initial_covariances = gmm.final_means_covariances(
    meg_data.subjects[0],
    n_states,
    gmm_kwargs={
        "n_init": 1,
        "verbose": 2,
        "verbose_interval": 50,
        "max_iter": 10000,
        "tol": 1e-6,
    },
    retry_attempts=1,
    learn_means=False,
    random_seed=124,
)

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_means=initial_means,
    initial_covariances=initial_covariances,
    rnn_type=rnn_type,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    normalization_type=normalization_type,
    alpha_xform=alpha_xform,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    learning_rate=learning_rate,
)
model.summary()

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

print("Training model")
history = model.fit(
    training_dataset,
    epochs=n_epochs,
    save_best_after=n_epochs_annealing,
    save_filepath="/well/woolrich/shared/vrad/trained_models/simulation_example1/weights",
)

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
alpha = model.predict_states(prediction_dataset)[0]
inf_stc = states.time_courses(alpha)

# Find correspondance to ground truth state time courses
matched_sim_stc, matched_inf_stc = states.match_states(sim.state_time_course, inf_stc)

print("Dice coefficient:", metrics.dice_coefficient(matched_sim_stc, matched_inf_stc))

# Delete the temporary folder holding the data
meg_data.delete_dir()
