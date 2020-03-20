import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


def sampling(args):
    """Reparameterization trick: draw random variable from normal distribution (mu=0,sigma=1)

    # Arguments
        z_mu, z_log_sigma = paramters of variational distribution, Q(Z)

    # Returns
        z* = a sample from the variational dist.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    chans = tf.keras.backend.int_shape(z_mean)[2]

    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim, chans))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def my_beautiful_custom_loss(alpha_ast,
                             Y_portioned,
                             npriors,
                             nchans,
                             SL_tmp_cov_mat,
                             inference_mu,
                             inference_sigma,
                             model_mu,
                             model_sigma,
                             weight, mini_batch_length):
    # sl_tmp_cov_mat <- The basis set at the sensor level after projection through H

    # Input dimensions:
    # cov_arg inv: [batches, mini_batch_length,n_channels,n_channels]
    # Y_portioned: [batches, mini_batch_length,n_channels]
    # weight: scalar, KL annealing term

    covariance_matrix = SL_tmp_cov_mat.astype('float32')

    # Alphas need to end up being of dimension (?,mini_batch_length,n_priors,1,1),
    # and need to undergo softplus transformation:
    alpha_ext = tf.keras.backend.expand_dims(tf.keras.backend.expand_dims(
        tf.keras.activations.softplus(alpha_ast),
        axis=-1), axis=-1)

    # Covariance basis functions need to be of dimension [1,1, n_priors, sensors, sensors]
    covariance_ext = tf.reshape(covariance_matrix, (1, 1, npriors, nchans, nchans))

    # Do the multiplicative sum over the n_priors dimension:
    cov_arg = tf.reduce_sum(tf.multiply(alpha_ext, covariance_ext), 2)

    # Add a tiny bit of diagonal to the covariance to ensure invertability
    safety_add = (1e-8) * np.eye(nchans, nchans)
    # cov_arg = (cov_arg + tf.transpose(cov_arg,perm=[0,1,3,2]))*0.5
    cov_arg = cov_arg + safety_add

    # The Log-Likelihood is given by:
    # c −(m*0.5)*log|Σ|−0.5*∑(x-mu)^Tsigma^-1 (x-mu)
    # where:
    #   c is some constant
    #   Σ is the covariance matrix
    #   mu is the mean (in this case, zero),
    #   x are the observations, i.e. Y_portioned here.
    #   m is the number of observations = channels
    m = nchans
    inv_cov_arg = tf.linalg.inv(cov_arg)
    log_det = -0.5 * m * tf.linalg.logdet(cov_arg)

    # Y_portioned is [batches, mini_batch_length,n_channels], but we need it to be
    # [batches, mini_batch_length,1,n_channels]. This is an easy fix - just add an extra dimension

    Y_exp_dims = tf.expand_dims(Y_portioned, axis=2)

    # Try and do Y * inv_cov* Y^T
    attempt = -0.5 * tf.matmul(tf.matmul(Y_exp_dims, inv_cov_arg),
                               tf.transpose(Y_exp_dims, perm=[0, 1, 3, 2]))
    # print("Y*inv_cov*Y^T spits out a tensor of shape",attempt.shape)
    # print("but we need it to be compatible with log_det, which is of shape",log_det.shape)
    # print("Easily achieved with tf.squeeze!")

    LL = log_det + tf.squeeze(attempt)

    # Now the KL divergence:
    shifted_model_mu = tf.roll(model_mu, shift=1, axis=1)
    p = tfd.Normal(loc=shifted_model_mu,
                   scale=tf.exp(model_sigma))

    # For the model sigma term, we take the mean over batches and MBL, then tile into the shape that
    # we need
    test = tf.tile(tf.math.reduce_mean(tf.math.reduce_mean(inference_sigma, axis=0), axis=0, keepdims=True),
                   [mini_batch_length, 1])

    q = tfd.Normal(loc=inference_mu,
                   scale=tf.exp(test))

    KL = tf.reduce_mean(tfp.distributions.kl_divergence(p, q))

    # loss = NLL + KL
    loss = tf.reduce_mean(-LL) + (weight * KL)

    return loss


def get_callbacks(callbacks):

    if isinstance(callbacks, str):
        callbacks = [callbacks]

    filepath = "model.h5"

    callback_dict = dict(
        # Early stopping:
        early_stopping=tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=10000,
                                                        restore_best_weights=True),

        # Decrease learning rate if we need to:
        reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                       factor=0.5,
                                                       min_lr=1e-6,
                                                       patience=40,
                                                       verbose=1),

        # Save the model as we train
        save_model=tf.keras.callbacks.ModelCheckpoint(filepath,
                                                      monitor='loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_freq=10),

        # NaN stopper
        nan_stop=tf.keras.callbacks.TerminateOnNaN()
    )

    return [callback_dict[callback] for callback in callbacks]
