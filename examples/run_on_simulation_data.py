"""Example script for running inference on simulated HMM data.

- Takes approximately 2 minutes to train (on compG017).
- Achieves a dice coefficient of ~0.9.
- Line 133 can be uncommented to produce a plot of the simulated and inferred
  state time courses for comparison.
"""

print("Importing packages")
import numpy as np
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad import array_ops, data
from vrad.inference import gmm, metrics, tf_ops
from vrad.inference.models.variational_rnn_autoencoder import create_model
from vrad.simulation import HMMSimulation
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

multi_gpu = False
strategy = None

# Settings
n_samples = 50000
observation_error = 0.2

n_states = 5
sequence_length = 100
batch_size = 32

learning_rate = 0.01
clip_normalization = None

do_annealing = True
annealing_sharpness = 5

n_epochs = 100
n_epochs_annealing = 80
n_epochs_burnin = 10

dropout_rate_inference = 0.4
dropout_rate_model = 0.4

n_units_inference = 64
n_units_model = 64

learn_means = False
learn_covariances = True

activation_function = "softmax"

# Load state transition probability matrix and covariances of each state
init_trans_prob = np.load("data/prob_000.npy")
init_djs = np.load("data/state_000.npy")

# Simulate data
print("Simulating data")
sim = HMMSimulation(
    trans_prob=init_trans_prob,
    djs=init_djs,
    n_samples=n_samples,
    e_std=observation_error,
)
meg_data = data.Data(sim)
n_channels = meg_data.shape[1]

# Priors
covariances, means = gmm.learn_mu_sigma(
    meg_data,
    n_states,
    take_random_sample=50000,
    gmm_kwargs={
        "n_init": 1,
        "verbose": 2,
        "verbose_interval": 50,
        "max_iter": 10000,
        "tol": 1e-6,
    },
    retry_attempts=5,
    learn_means=False,
)

# Prepare dataset
training_dataset, prediction_dataset = tf_ops.train_predict_dataset(
    time_series=meg_data, sequence_length=sequence_length, batch_size=batch_size,
)

# Build autoecoder model
rnn_vae = create_model(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_mean=means,
    initial_covariances=covariances,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    activation_function=activation_function,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    n_epochs_burnin=n_epochs_burnin,
    learning_rate=learning_rate,
    clip_normalization=clip_normalization,
    multi_gpu=multi_gpu,
    strategy=strategy,
)

rnn_vae.summary()

# Train the model
print("Training model")
history = rnn_vae.fit(
    training_dataset,
    callbacks=[TqdmCallback(tqdm_class=tqdm, verbose=0)],
    epochs=n_epochs,
    verbose=0,
)

# Inferred state time course
inf_stc = array_ops.get_one_hot(
    np.concatenate(rnn_vae.predict(prediction_dataset)["m_theta_t"]).argmax(axis=1)
)

# Find correspondance to ground truth state time courses
matched_stc, matched_inf_stc = array_ops.match_states(sim.state_time_course, inf_stc)

# Compare state time courses
# plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))
