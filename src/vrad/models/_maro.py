"""Class for a Gaussian observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers, optimizers
from vrad import models
from vrad.inference import initializers, losses
from vrad.models.layers import LogLikelihoodLayer, MARMeanCovLayer, MARParametersLayer
from vrad.utils.misc import replace_argument


class MARO(models.Base):
    """Multivariate Autoregressive Observations (MARO) model.

    Parameters
    ----------
    dimensions : vrad.models.config.Dimensions
        Dimensions of data in the model.
    observation_model : vrad.models.config.ObservationModel
        Parameters related to the observation model.
    training : vrad.models.config.Training
        Parameters related to training a model.
    """

    def __init__(
        self,
        dimensions: models.config.Dimensions,
        observation_model: models.config.ObservationModel,
        training: models.config.Training,
    ):
        if observation_model.model != "multivariate_autoregressive":
            raise ValueError("Observation model must be multivariate_autoregressive.")

        # The base class will build and compile the keras model
        super().__init__(dimensions, observation_model, training)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.dimensions, self.observation_model)

    def compile(self):
        """Wrapper for the standard keras compile method."""
        self.model.compile(
            optimizer=self.training.optimizer, loss=[losses.ModelOutputLoss()]
        )

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
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks=[],
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

    def set_params(self, coeffs, cov):
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
    dimensions: models.config.Dimensions,
    observation_model: models.config.ObservationModel,
):
    """Model structure.

    Parameters
    ----------
    dimensions : vrad.models.config.Dimensions
    observation_model : vrad.models.config.ObservationModel

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layers for inputs
    data = layers.Input(
        shape=(dimensions.sequence_length, dimensions.n_channels), name="data"
    )
    alpha = layers.Input(
        shape=(dimensions.sequence_length, dimensions.n_states), name="alpha"
    )

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma_t), where
    #      - mu_t = Sum_j Sum_l alpha_jt coeffs_jt data_{t-l}.
    #      - sigma_t = Sum_j alpha^2_jt cov_j, where cov_j is a learnable
    #        diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with alpha_jt
    #   and the observation model.

    # Definition of layers
    mar_params_layer = MARParametersLayer(
        dimensions.n_states,
        dimensions.n_channels,
        observation_model.n_lags,
        observation_model.initial_coeffs,
        observation_model.initial_cov,
        observation_model.learn_coeffs,
        observation_model.learn_cov,
        name="mar_params",
    )
    mean_cov_layer = MARMeanCovLayer(
        dimensions.n_states,
        dimensions.n_channels,
        dimensions.sequence_length,
        observation_model.n_lags,
        name="mean_cov",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    coeffs, cov = mar_params_layer(data)  # data not used
    clipped_data, mu, sigma = mean_cov_layer([data, alpha, coeffs, cov])
    ll_loss = ll_loss_layer([clipped_data, mu, sigma])

    return Model(inputs=[data, alpha], outputs=[ll_loss])
