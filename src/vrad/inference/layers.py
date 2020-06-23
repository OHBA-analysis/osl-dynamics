from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from vrad.inference.inference_functions import pseudo_sigma_to_sigma
from vrad.inference.initializers import Identity3D


def sampling(args):
    """Reparameterization trick:
     draw random variable from normal distribution (mu=0, sigma=1)

     Arguments:
     - z_mu, z_log_sigma = paramters of variational distribution, Q(Z)

     Returns:
     - z* = a sample from the variational distribution.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    chans = K.int_shape(z_mean)[2]
    epsilon = K.random_normal(shape=(batch, dim, chans))
    # default mean=0, std=1
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ReparameterizationLayer(Layer):
    """Resample a normal distribution.

    The reparameterization trick is used to provide a differentiable random sample.
    By optimizing the values which define the distribution (i.e. mu and sigma), a model
    can be trained while still maintaining its stochasticity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: Union[tf.Tensor, tf.Tensor], **kwargs):
        """Given the mean and log standard deviation, create a normal distribution.

        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.Tensor]
            Mean and log of standard deviation.
        kwargs

        Returns
        -------
        new_distribution : tf.Tensor
            The newly sampled normal distribution created from its mean and the log
            of its standard deviation.
        """
        z_mean, z_log_var = inputs
        new_distribution = tf.random.normal(
            z_mean.shape, mean=z_mean, stddev=tf.exp(0.5 * z_log_var)
        )
        return new_distribution


class MVNLayer(Layer):
    """Parameterises multiple multivariate Gaussians and in terms of their means
       and covariances. Means and covariances are outputted.
    """

    def __init__(
        self,
        n_gaussians,
        dim,
        learn_means=True,
        learn_covs=True,
        initial_means=None,
        initial_pseudo_sigmas=None,
        **kwargs
    ):
        super(MVNLayer, self).__init__(**kwargs)
        self.n_gaussians = n_gaussians
        self.dim = dim
        self.learn_means = learn_means
        self.learn_covs = learn_covs
        self.initial_means = initial_means
        self.burnin = tf.Variable(False, trainable=False)

        # Only keep the lower triangle of pseudo sigma (also flattens the tensors)
        self.initial_pseudo_sigmas = tfp.math.fill_triangular_inverse(
            initial_pseudo_sigmas
        ).numpy()

    def _means_initializer(self, shape, dtype=tf.float32):
        mats = self.initial_means
        assert (
            len(shape) == 2 and shape[0] == mats.shape[0] and shape[1] == mats.shape[1]
        ), "shape must be (N, D)"
        return mats

    def _pseudo_sigmas_initializer(self, shape, dtype=tf.float32):
        """Initialize a stacked tensor of using self.initial_pseudo_sigmas."""
        return self.initial_pseudo_sigmas

    def build(self, input_shape):
        # Initialiser for means
        if self.initial_means is None:
            self.means_initializer = tf.keras.initializers.Zeros
        else:
            self.means_initializer = self._means_initializer

        self.means = self.add_weight(
            "means",
            shape=(self.n_gaussians, self.dim),
            dtype=tf.float32,
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )

        # Initialiser for covs
        if self.initial_pseudo_sigmas is None:
            # CHANGED THE IMPLEMENTATION SUCH THAT PSEUDO SIGMAS ARE FLATTENED
            # SO THIS WONT WORK ANYMORE
            print("NEED TO CODE: see MVNLayer")
            exit()
            self.pseudo_sigmas_initializer = Identity3D
        else:
            self.pseudo_sigmas_initializer = self._pseudo_sigmas_initializer

        self.pseudo_sigmas = self.add_weight(
            "pseudo_sigmas",
            shape=self.initial_pseudo_sigmas.shape,
            dtype=tf.float32,
            initializer=self.pseudo_sigmas_initializer,
            trainable=self.learn_covs,
        )

        self.untrainable_sigmas = self.add_weight(
            "pseudo_sigmas",
            shape=self.initial_pseudo_sigmas.shape,
            dtype=tf.float32,
            initializer=self.pseudo_sigmas_initializer,
            trainable=False,
        )

        self.built = True

    def call(self, inputs, **kwargs):

        self.sigmas = tf.cond(
            self.burnin,
            lambda: pseudo_sigma_to_sigma(self.untrainable_sigmas),
            lambda: pseudo_sigma_to_sigma(self.pseudo_sigmas),
        )
        self.sigmas = pseudo_sigma_to_sigma(self.pseudo_sigmas)
        return self.means, self.sigmas

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([self.n_gaussians, self.dim]),
            tf.TensorShape([self.n_gaussians, self.dim, self.dim]),
        ]

    def get_config(self):
        config = super(MVNLayer, self).get_config()
        config.update(
            {
                "n_gaussian": self.n_gaussians,
                "dim": self.dim,
                "learn_means": self.learn_means,
                "learn_covs": self.learn_covs,
                "initial_means": self.initial_means,
                "initial_pseudo_sigmas": self.initial_pseudo_sigmas,
            }
        )
        return config


class TrainableVariablesLayer(Layer):
    """Generic trainable variables layer.

       Sets up trainable parameters/weights tensor of a certain shape.
       Parameters/weights are outputted.
    """

    def __init__(self, shape, initial_values=None, trainable=True, **kwargs):
        super(TrainableVariablesLayer, self).__init__(**kwargs)
        self.shape = shape
        self.initial_values = initial_values
        self.trainable = trainable

    def _variables_initializer(self, shape, dtype=tf.float32):
        values = self.initial_values
        return values

    def build(self, input_shape):
        # Set initialiser for means
        if self.initial_values is None:
            self.values_initializer = tf.keras.initializers.Zeros
        else:
            self.values_initializer = self._variables_initializer

        self.values = self.add_weight(
            "values",
            shape=self.shape,
            dtype=K.floatx(),
            initializer=self.values_initializer,
            trainable=self.trainable,
        )

        self.built = True

    def call(self, inputs, **kwargs):
        return self.values

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.shape)

    def get_config(self):
        config = super(TrainableVariablesLayer, self).get_config()
        config.update(
            {
                "shape": self.shape,
                "initial_values": self.initial_values,
                "trainable": self.trainable,
            }
        )
        return config
