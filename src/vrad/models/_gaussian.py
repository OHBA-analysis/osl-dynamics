"""Class for a Gaussian observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers, optimizers
from tensorflow.nn import softplus
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    trace_normalize,
)
from vrad.models import BaseModel
from vrad.models.layers import MeansCovsLayer, MixMeansCovsLayer, LogLikelihoodLayer
from vrad.inference.losses import LogLikelihoodLoss


class Gaussian(BaseModel):
    """Generative model with Gaussian observations.

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
    do_annealing : bool
        Should we use KL annealing during training?
    annealing_sharpness : float
        Parameter to control the annealing curve.
    n_epochs_annealing : int
        Number of epochs to perform annealing.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_alpha_scaling: bool,
        normalize_covariances: bool,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_covariances: np.ndarray = None,
    ):
        # Parameters related to the observation model
        self.learn_alpha_scaling = learn_alpha_scaling
        self.normalize_covariances = normalize_covariances
        self.initial_covariances = initial_covariances

        # Initialise the model base class
        # This will build and compile the keras model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            do_annealing=do_annealing,
            annealing_sharpness=annealing_sharpness,
            n_epochs_annealing=n_epochs_annealing,
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
        cholesky_covariances = means_covs_layer.cholesky_covariances
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

        # Replace covariances in the layer weights
        for i in range(len(layer_weights)):
            if layer_weights[i].shape == covariances.shape:
                layer_weights[i] = cholesky_factor(covariances)

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
        learn_covariances=True,
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
