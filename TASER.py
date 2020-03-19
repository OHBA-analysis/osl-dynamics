import numpy as np
import scipy.io as spio
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import ops

import importlib

import model_functions
model_functions = importlib.reload(model_functions)

tf.random.set_seed(42)
np.random.seed(154)

tfd = tfp.distributions
tf.config.experimental_run_functions_eagerly(True)
ops.reset_default_graph()

base_dir = '/well/woolrich/users/ozn760/TASER/bSNR_1_iteration_1/'

# Load in PCA'd lead field, the GT X and simulated data, Y
H_PCA = spio.loadmat(base_dir + 'burst_SIMULATION_LF_PCA.mat')
H_PCA = np.transpose(H_PCA['c_ROI_LF_PCA'])  # Must be channels or PCs by sources

Y_sim = spio.loadmat(base_dir + 'burst_SIMULATION_Y_PCA.mat')
Y_sim = Y_sim['data_for_MF_SR']  # needs to be channels or PCs by time
Y_sim = Y_sim[:, :]
print("Simulated data are of dimension:", Y_sim.shape)

H2 = H_PCA

trial_times = spio.loadmat(base_dir + 'trial_times.mat')
trial_times = trial_times['trial_times']

# Finally, load in the ROB prior:
ROB = spio.loadmat(base_dir + 'ROB_prior.mat')
ROB = ROB['ROB_prior']

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

model = model_functions.create_model(mini_batch_length, nchans, npriors, SL_tmp_cov_mat)

# Early stopping:
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=10000,
                                                          restore_best_weights=True)

# Decrease learning rate if we need to:
callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                          factor=0.5,
                                                          min_lr=1e-6,
                                                          patience=40,
                                                          verbose=1)

# Save the model as we train
filepath = "model.h5"
save_model = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                period=10)

# NaN stopper
NaNstop = tf.keras.callbacks.TerminateOnNaN()

# Y_portioned is of shape ((nbatches,mini_batch_length,nchans))
ops.reset_default_graph()
history = model.fit(Y_portioned,  # Input or "Y_true"
                    verbose=1,
                    callbacks=[earlystopping_callback, callback_reduce_lr],
                    shuffle=True,
                    epochs=30,
                    batch_size=10,  # Scream if you want to go faster
                    )
