"""Class for a Gaussian observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers, optimizers
from vrad import models
from vrad.inference.losses import LogLikelihoodLoss
from vrad.models.layers import LogLikelihoodLayer, MARParametersLayer, MARMeanCovLayer
from vrad.utils.misc import replace_argument


class MARO(models.Base):
    """Multivariate Autoregressive Observations (MARO) model.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    n_lags : int
        Order of the multivariate autoregressive observation model.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training? Optional.
    strategy : str
        Strategy for distributed learning. Optional.
    initial_coeffs : np.ndarray
        Initial values for the MAR coefficients. Optional.
    initial_cov : np.ndarray
        Initial values for the covariances. Optional.
    learn_coeffs : bool
        Should we learn the MAR coefficients? Optional, default is True.
    learn_cov : bool
        Should we learn the covariances. Optional, default is True.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        n_lags: int,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_coeffs: np.ndarray = None,
        initial_cov: np.ndarray = None,
        learn_coeffs: bool = True,
        learn_cov: bool = True,
    ):
        # Parameters related to the observation model
        self.n_lags = n_lags
        self.initial_coeffs = initial_coeffs
        self.initial_cov = initial_cov
        self.learn_coeffs = learn_coeffs
        self.learn_cov = learn_cov

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
            n_lags=self.n_lags,
            initial_coeffs=self.initial_coeffs,
            initial_cov=self.initial_cov,
            learn_coeffs=self.learn_coeffs,
            learn_cov=self.learn_cov,
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

    def get_params(self):
        """Get the parameters of the MAR model.

        Returns
        -------
        coeffs : np.ndarray
            MAR coefficients. Shape is (n_states, n_lags, n_channels, n_channels).
        cov : np.ndarray
            Mar covariance. Shape is (n_states, n_channels, n_channels).
        """
        mar_params_layer = self.model.get_layer("mar_params")
        coeffs = mar_params_layer.coeffs.numpy()
        cov = np.array([np.diag(c) for c in mar_params_layer.cov.numpy()])
        return coeffs, cov

    def set_params(self):
        """Set the parameters of the MAR model.

        Parameters
        ----------
        coeffs : np.ndarray
            MAR coefficients. Shape is (n_states, n_lags, n_channels, n_channels).
        cov : np.ndarray
            Mar covariance. Shape is (n_states, n_channels, n_channels).
        """
        mar_params_layer = self.model.get_layer("mar_params")
        layer_weights = mar_params_layer.get_weights()
        cov = np.array([np.diag(c) for c in cov])
        for i in range(len(layer_weights)):
            if layer_weights[i].shape == coeffs.shape:
                layer_weights[i] = coeffs
            if layer_weights[i].shape == cov.shape:
                layer_weights[i] = cov
        mar_params_layer.set_weights(layer_weights)


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_lags: int,
    initial_coeffs: np.ndarray,
    initial_cov: np.ndarray,
    learn_coeffs: bool,
    learn_cov: bool,
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
    n_lags : int
        Order of the multivariate autoregressive observation model.
    initial_coeffs : np.ndarray
        Initial values for the MAR coefficients.
    initial_cov : np.ndarray
        Initial values for the covariances.
    learn_coeffs : bool
        Should we learn the MAR coefficients?
    learn_cov : bool
        Should we learn the covariances.

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layers for inputs
    data_t = layers.Input(shape=(sequence_length, n_channels), name="data")
    alpha_jt = layers.Input(shape=(sequence_length, n_states), name="alpha_t")

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma_t), where
    #      - mu_t = Sum_j Sum_l alpha_jt coeffs_jt data_{t-l}.
    #      - sigma_t = Sum_j alpha^2_jt cov_j, where cov_j is a learnable
    #        diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with alpha_jt
    #   and the observation model.

    # Definition of layers
    mar_params_layer = MARParametersLayer(
        n_states,
        n_channels,
        n_lags,
        initial_coeffs,
        initial_cov,
        learn_coeffs,
        learn_cov,
        name="mar_params",
    )
    mean_cov_layer = MARMeanCovLayer(
        n_states, n_channels, sequence_length, n_lags, name="mean_cov"
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    coeffs_jl, cov_j = mar_params_layer(data_t)  # data_t not used
    clipped_data_t, mu_t, sigma_t = mean_cov_layer([data_t, alpha_jt, coeffs_jl, cov_j])
    ll_loss = ll_loss_layer([clipped_data_t, mu_t, sigma_t])

    return Model(inputs=[data_t, alpha_jt], outputs=[ll_loss])
