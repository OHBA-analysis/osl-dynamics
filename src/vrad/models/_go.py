"""Class for a Gaussian observation model.

"""

import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers, optimizers
from tensorflow.nn import softplus
from vrad import models
from vrad.inference import initializers
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    trace_normalize,
)
from vrad.inference.losses import LogLikelihoodLoss
from vrad.models.layers import LogLikelihoodLayer, MeansCovsLayer, MixMeansCovsLayer
from vrad.utils.misc import replace_argument


class GO(models.Base):
    """Gaussian Observations (GO) model.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels). Optional.
    learn_covariances : bool
        Should we learn the covariance matrix for each state?
        Optional, default is True.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_alpha_scaling: bool,
        normalize_covariances: bool,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_covariances: np.ndarray = None,
        learn_covariances: bool = True,
    ):
        # Parameters related to the observation model
        self.learn_alpha_scaling = learn_alpha_scaling
        self.normalize_covariances = normalize_covariances
        self.initial_covariances = initial_covariances
        self.learn_covariances = learn_covariances

        # Initialise the model base class
        # This will build and compile the keras model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            multi_gpu=multi_gpu,
            strategy=strategy,
        )

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(
            n_states=self.n_states,
            n_channels=self.n_channels,
            sequence_length=self.sequence_length,
            learn_alpha_scaling=self.learn_alpha_scaling,
            normalize_covariances=self.normalize_covariances,
            initial_covariances=self.initial_covariances,
            learn_covariances=self.learn_covariances,
        )

    def compile(self):
        """Wrapper for the standard keras compile method.

        Sets up the optimizer and loss functions.
        """
        # Setup optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # Loss functions
        ll_loss = LogLikelihoodLoss()

        # Compile
        self.model.compile(optimizer=optimizer, loss=[ll_loss])

    def fit(
        self,
        *args,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                True,
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weights(self):
        """Reset the model as if you've built a new model."""
        self.compile()
        initializers.reinitialize_model_weights(self.model)

    def get_covariances(self, alpha_scale=True):
        """Get the covariances of each state.

        Parameters
        ----------
        alpah_scale : bool
            Should we apply alpha scaling? Default is True.

        Returns
        -------
        np.ndarary
            State covariances.
        """
        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        cholesky_covariances = tfp.math.fill_triangular(
            means_covs_layer.flattened_cholesky_covariances
        )
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances).numpy()

        # Normalise covariances
        if self.normalize_covariances:
            covariances = trace_normalize(covariances).numpy()

        # Apply alpha scaling
        if alpha_scale:
            alpha_scaling = self.get_alpha_scaling()
            covariances *= alpha_scaling[:, np.newaxis, np.newaxis]

        return covariances

    def set_covariances(self, covariances):
        """Set the covariances of each state.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances.
        """
        means_covs_layer = self.model.get_layer("means_covs")
        layer_weights = means_covs_layer.get_weights()

        flattened_covariances_shape = (
            covariances.shape[0],
            covariances.shape[1] * (covariances.shape[1] + 1) // 2,
        )

        # Replace covariances in the layer weights
        for i in range(len(layer_weights)):
            if layer_weights[i].shape == flattened_covariances_shape:
                cholesky_covariances = cholesky_factor(covariances)
                flattened_cholesky_covariances = tfp.math.fill_triangular_inverse(
                    cholesky_covariances
                )
                layer_weights[i] = flattened_cholesky_covariances

        # Set the weights of the layer
        means_covs_layer.set_weights(layer_weights)

    def get_alpha_scaling(self):
        """Get the alpha scaling of each state.

        Returns
        ----------
        bool
            Alpha scaling for each state.
        """
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        alpha_scaling = softplus(alpha_scaling).numpy()
        return alpha_scaling


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    learn_alpha_scaling: bool,
    normalize_covariances: bool,
    initial_covariances: np.ndarray,
    learn_covariances: bool,
):
    """Model structure.

    Parameters
    ----------
    n_states : int
        Numeber of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    learn_covariances : bool
        Should we learn the covariances?

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layers for inputs
    data = layers.Input(shape=(sequence_length, n_channels), name="data")
    alpha_t = layers.Input(shape=(sequence_length, n_states), name="alpha_t")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha_t
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        n_states,
        n_channels,
        learn_means=False,
        learn_covariances=learn_covariances,
        normalize_covariances=normalize_covariances,
        initial_means=None,
        initial_covariances=initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, learn_alpha_scaling, name="mix_means_covs"
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu_j, D_j = means_covs_layer(data)  # data not used
    m_t, C_t = mix_means_covs_layer([alpha_t, mu_j, D_j])
    ll_loss = ll_loss_layer([data, m_t, C_t])

    return Model(inputs=[data, alpha_t], outputs=[ll_loss])
