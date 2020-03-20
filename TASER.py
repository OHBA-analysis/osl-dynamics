import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

import importlib

import model_functions
import plotting_functions
import misc
import taser_functions

model_functions = importlib.reload(model_functions)
plotting_functions = importlib.reload(plotting_functions)
misc = importlib.reload(misc)
taser_functions = importlib.reload(taser_functions)

log_print = misc.log_print

log_print("Imports completed", "blue")

tf.random.set_seed(42)
np.random.seed(154)

tfd = tfp.distributions
tf.config.experimental_run_functions_eagerly(False)
ops.reset_default_graph()

base_dir = '/well/woolrich/users/ozn760/TASER/bSNR_1_iteration_1/'

file_names = dict(
    H_PCA=base_dir + 'burst_SIMULATION_LF_PCA.npy',
    Y_sim=base_dir + 'burst_SIMULATION_Y_PCA.npy',
    trial_times=base_dir + 'trial_times.npy',
    ROB_prior=base_dir + 'ROB_prior.npy'
)

# Load in PCA'd lead field, the GT X and simulated data, Y
log_print(f"Loading PCA'd lead field from {file_names['H_PCA']}", 'green')
H_PCA = np.load(file_names['H_PCA']).T
H2 = H_PCA

log_print(f"Loading PCA'd Y from {file_names['Y_sim']}", 'green')
Y_sim = np.load(file_names['Y_sim'])
log_print(f"Simulated data are of dimension: {Y_sim.shape}", "")

log_print(f"Loading trial times from {file_names['trial_times']}", 'green')
trial_times = np.load(file_names['trial_times'])

# Finally, load in the ROB prior:
log_print(f"Loading 'rest of brain' prior from {file_names['ROB_prior']}", 'green')
ROB = np.load(file_names['ROB_prior'])

log_print("Files loaded", 'blue')

print_simulation_info = False
if print_simulation_info:
    # Spit out simulation info
    with open(base_dir + 'exp.txt', 'r') as f:
        print(f.read())

n_batches = 198  # how many examples of data we have
mini_batch_length = 100  # how long each feature/segment/example is
n_channels = H2.shape[0]
n_sources = H2.shape[1]
n_units = n_sources  # number of GRU units
n_priors = n_sources + 2  # number of priors in our inference
tf.keras.backend.clear_session()

# Simulate data - not optimised!
time = np.arange(0, n_batches * mini_batch_length, 1);

# First, build the prior basis set.
tmp_cov_mat = np.zeros((n_priors, n_sources, n_sources))  # source level covariance
sl_tmp_cov_mat = np.zeros((n_priors, n_channels, n_channels))  # sensor level covariance
debug_cov = 0

for i in range(n_priors - 2):
    tmp_cov_mat[i, i, i] = 1
    sl_tmp_cov_mat[i, :, :] = (1e-4 * np.eye(n_channels)) + np.matmul(np.matmul(H2, tmp_cov_mat[i, :, :]),
                                                                      np.transpose(H2))  # C_Yi = H C_Xi H'

sl_tmp_cov_mat[n_priors - 2, :, :] = np.eye(n_channels)  # REGULARISATION TERM!
sl_tmp_cov_mat[n_priors - 1, :, :] = ROB

Y_portioned = np.reshape(Y_sim[:, 0:n_batches * mini_batch_length].transpose(),
                         [n_batches, mini_batch_length, n_channels])
print(Y_portioned.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = model_functions.create_model(mini_batch_length, n_channels, n_priors, sl_tmp_cov_mat)

    # Y_portioned is of shape ((n_batches,mini_batch_length,n_channels))
    # ops.reset_default_graph()

    # log_print(f"LOG: scope is {ops.get_default_graph()._distribution_strategy_stack}", "red")


history = model.fit(Y_portioned,  # Input or "Y_true"
                    verbose=1,
                    callbacks=taser_functions.get_callbacks(["early_stopping", "reduce_lr"]),
                    shuffle=True,
                    epochs=4,
                    batch_size=10,  # Scream if you want to go faster
                    )

log_print("Training complete", "green")

mu_store = np.zeros((n_batches, mini_batch_length, n_priors))
for i in range(n_batches):
    mu, sigma, ast, mod_mu, mod_sigma = model.predict(Y_portioned[i:i + 1, :, :])
    mu_store[i, :, :] = mu

print(mu_store.shape)
mu_rs = tf.math.softplus(np.reshape(mu_store, (n_batches * mini_batch_length, n_priors)))
mu_rs = np.array(mu_rs)

loss = np.array(history.history['loss'])

plotting = True
plotting_functions.plot_all(loss, trial_times, mu_rs, n_priors)

np.save(base_dir + 'alphas.npy', mu_rs)
np.save(base_dir + 'loss.npy', loss)
