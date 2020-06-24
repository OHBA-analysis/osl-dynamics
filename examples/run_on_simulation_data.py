print("Importing packages")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad import array_ops, data
from vrad.inference import gmm, metrics, tf_ops
from vrad.inference.models.variational_rnn_autoencoder import create_model
from vrad.simulation import HMMSimulation
from vrad.utils import misc, plotting

# Limit GPU RAM use
tf_ops.gpu_growth()

# Parameters
n_states = 5
sequence_length = 100
batch_size = 32
learning_rate = 0.01

do_annealing = True
annealing_sharpness = 5
n_epochs_annealing = 80

n_epochs = 100
n_epochs_burnin = 10

dropout_rate_encoder = 0.4
dropout_rate_decoder = 0.4

n_units_encoder = 64
n_units_decoder = 64

learn_means = False
learn_covs = True

activation_function = "softmax"

# Load state transition probability matrix and covariances of each state
init_trans_prob = np.load("data/prob_000.npy")
init_djs = np.load("data/state_000.npy")

# Simulate data
print("Simulating data")
sim = HMMSimulation(
    trans_prob=init_trans_prob, djs=init_djs, n_samples=50000, e_std=0.2
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

covariances = gmm.find_cholesky_decompositions(covariances, means, False)

# Prepare dataset
training_dataset, prediction_dataset = tf_ops.train_predict_dataset(
    time_series=meg_data, sequence_length=sequence_length, batch_size=batch_size,
)

# Build autoecoder model
rnn_vae = create_model(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=False,
    learn_covs=True,
    initial_mean=means,
    initial_cholesky_cov=covariances,
    n_units_encoder=n_units_encoder,
    n_units_decoder=n_units_decoder,
    dropout_rate_encoder=dropout_rate_encoder,
    dropout_rate_decoder=dropout_rate_decoder,
    activation_function=activation_function,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    n_epochs_burnin=n_epochs_burnin,
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

# Evaluate performance
inf_stc = array_ops.get_one_hot(
    np.concatenate(rnn_vae.predict(prediction_dataset)[3]).argmax(axis=1)
)

matched_stc, matched_inf_stc = array_ops.match_states(sim.state_time_course, inf_stc)

plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))
