import numpy as np
import scipy.io as spio
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
tf.config.experimental_run_functions_eagerly(True)
ops.reset_default_graph()

base_dir = '/well/woolrich/users/ozn760/TASER/bSNR_1_iteration_1/'

# Load in PCA'd lead field, the GT X and simulated data, Y
log_print(f"Loading PCA'd lead field from {base_dir + 'burst_SIMULATION_LF_PCA.mat'}", 'green')
H_PCA = spio.loadmat(base_dir + 'burst_SIMULATION_LF_PCA.mat')
H_PCA = np.transpose(H_PCA['c_ROI_LF_PCA'])  # Must be channels or PCs by sources

log_print(f"Loading PCA'd Y from {base_dir + 'burst_SIMULATION_Y_PCA.mat'}", 'green')
Y_sim = spio.loadmat(base_dir + 'burst_SIMULATION_Y_PCA.mat')
Y_sim = Y_sim['data_for_MF_SR']  # needs to be channels or PCs by time
Y_sim = Y_sim[:, :]
log_print(f"Simulated data are of dimension: {Y_sim.shape}", "magenta")

H2 = H_PCA

log_print(f"Loading trial times from {base_dir + 'trial_times.mat'}", 'green')
trial_times = spio.loadmat(base_dir + 'trial_times.mat')
trial_times = trial_times['trial_times']

# Finally, load in the ROB prior:
log_print(f"Loading 'rest of brain' prior from {base_dir + 'ROB_prior.mat'}", 'green')
ROB = spio.loadmat(base_dir + 'ROB_prior.mat')
ROB = ROB['ROB_prior']

log_print("Files loaded", 'blue')

print_simulation_info = False
if print_simulation_info:
    # Spit out simulation info
    with open(base_dir + 'exp.txt', 'r') as f:
        print(f.read())

nbatches = 198  # how many examples of data we have
mini_batch_length = 100  # how long each feature/segment/example is
nchans = H2.shape[0]
nsources = H2.shape[1]
nunits = nsources  # number of GRU units
npriors = nsources + 2  # number of priors in our inference
tf.keras.backend.clear_session()

# Simulate data - not optimised!
time = np.arange(0, nbatches * mini_batch_length, 1);

# First, build the prior basis set.
tmp_cov_mat = np.zeros((npriors, nsources, nsources))  # source level covariance
SL_tmp_cov_mat = np.zeros((npriors, nchans, nchans))  # sensor level covariance
debug_cov = 0

for i in range(npriors - 2):
    tmp_cov_mat[i, i, i] = 1
    SL_tmp_cov_mat[i, :, :] = (1e-4 * np.eye(nchans)) + np.matmul(np.matmul(H2, tmp_cov_mat[i, :, :]),
                                                                  np.transpose(H2))  # C_Yi = H C_Xi H'

SL_tmp_cov_mat[npriors - 2, :, :] = np.eye(nchans)  # REGULARISATION TERM!
SL_tmp_cov_mat[npriors - 1, :, :] = ROB

Y_portioned = np.reshape(Y_sim[:, 0:nbatches * mini_batch_length].transpose(),
                         [nbatches, mini_batch_length, nchans])
print(Y_portioned.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = model_functions.create_model(mini_batch_length, nchans, npriors, SL_tmp_cov_mat)

    # Y_portioned is of shape ((nbatches,mini_batch_length,nchans))
    # ops.reset_default_graph()

    log_print(f"LOG: scope is {ops.get_default_graph()._distribution_strategy_stack}", "red")
    history = model.fit(Y_portioned,  # Input or "Y_true"
                        verbose=1,
                        callbacks=taser_functions.get_callbacks(["early_stopping", "reduce_lr"]),
                        shuffle=True,
                        epochs=1,
                        batch_size=10,  # Scream if you want to go faster
                        )

log_print("Training complete", "green")

mu_store = np.zeros((nbatches, mini_batch_length, npriors))
for i in range(nbatches):
    mu, sigma, ast, mod_mu, mod_sigma = model.predict(Y_portioned[i:i + 1, :, :])
    mu_store[i, :, :] = mu

print(mu_store.shape)
mu_rs = tf.math.softplus(np.reshape(mu_store, (nbatches * mini_batch_length, npriors)))
mu_rs = np.array(mu_rs)

loss = np.array(history.history['loss'])

plotting = True
if plotting:
    print("Plotting results")
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    axis_generator = (axis for axis in axes.flatten())

    plotting_functions.plot(np.log(loss - loss.min() + 1e-6), title="adjusted log loss", axis=next(axis_generator))

    plotting_functions.imshow(trial_times[1:1000, :].T, title="trial_times", axis=next(axis_generator))
    plotting_functions.imshow(mu_rs[1:1000, 88:100].T, title="Recon alphas", axis=next(axis_generator))
    plotting_functions.imshow(mu_rs[1:1000, :].T, title="Recon alphas", axis=next(axis_generator))

    plotting_functions.plot_alpha_channel(mu_rs, channel=88, axis=next(axis_generator))
    plotting_functions.plot_alpha_channel(mu_rs, channel=89, axis=next(axis_generator))
    plotting_functions.plot_alpha_channel(mu_rs, channel=98, axis=next(axis_generator))

    plotting_functions.plot_multiple_channels(mu_rs, channels=[89, 98], axis=next(axis_generator))

    plotting_functions.plot_reg(mu_rs, npriors=npriors, axis=next(axis_generator))

    plt.tight_layout()

    plt.show()

np.save(base_dir + 'alphas.npy', mu_rs)
np.save(base_dir + 'loss.npy', loss)
