"""Single-channel Dynamic Network Modes (SC-DyNeMo)."""

from dataclasses import dataclass

import tensorflow as tf

from osl_dynamics.models.dynemo import Config as DynemoConfig
from osl_dynamics.models.dynemo import Model as DynemoModel
from osl_dynamics.inference.layers import (
    OscillatorCovarianceMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
)


@dataclass
class Config(DynemoConfig):
    """Additional parameters for SC-DyNeMo.

    Parameters
    ----------
    sampling_frequency : float
        The sampling frequency of the data (Hz).
    frequency_range : tuple[float, float]
        Limits for the frequency parameter.
        Upper limit should not be higher than the Nyquist frequency.
    """

    model_name: str = "SC-DyNeMo"

    sampling_frequency: float = None
    frequency_range: list = None

    def __post_init__(self):
        super().__post_init__()


class Model(DynemoModel):
    r"""Single-channel Dynamic Network Modes (SC-DyNeMo).

    This model is a single-channel version of DyNeMo.
    It should only be used for single-channel data which has been time-embedded.

    This model parameterises the covariance matrice for each model assuming a
    stochastic oscillators.

    The parameters are the amplitude (:math:`A`), frequency (:math:`f`), and
    variance of added Gaussian noise (:math:`\sigma^2`) of oscillators. The
    parameters define the auto-covariance matrix as:

    .. math::
        C_{ij} = \frac{1}{2} A^2 \cos(2 \pi f \Delta t) + \delta_{ij} \sigma^2

    Parameters
    ----------
    config : Config
        The model configuration.
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""

        config = self.config

        # ---------- Define layers ---------- #
        data_drop_layer = tf.keras.layers.Dropout(
            config.inference_dropout, name="data_drop"
        )
        inf_rnn_layer = InferenceRNNLayer(
            config.inference_rnn,
            config.inference_normalization,
            config.inference_activation,
            config.inference_n_layers,
            config.inference_n_units,
            config.inference_dropout,
            config.inference_regularizer,
            name="inf_rnn",
        )
        inf_mu_layer = tf.keras.layers.Dense(config.n_modes, name="inf_mu")
        inf_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="inf_sigma"
        )
        theta_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon,
            name="theta",
        )
        alpha_layer = SoftmaxLayer(
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="alpha",
        )
        means_layer = VectorsLayer(
            config.n_modes,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        covs_layer = OscillatorCovarianceMatricesLayer(
            n=self.config.n_modes,
            m=self.config.n_channels,
            sampling_frequency=self.config.sampling_frequency,
            frequency_range=self.config.frequency_range,
            learn=self.config.learn_covariances,
            epsilon=self.config.covariances_epsilon,
            name="covs",
        )
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_covs_layer = MixMatricesLayer(name="mix_covs")
        ll_loss_layer = LogLikelihoodLossLayer(config.loss_calc, name="ll_loss")
        theta_drop_layer = tf.keras.layers.Dropout(
            config.model_dropout,
            name="theta_drop",
        )
        mod_rnn_layer = ModelRNNLayer(
            config.model_rnn,
            config.model_normalization,
            config.model_activation,
            config.model_n_layers,
            config.model_n_units,
            config.model_dropout,
            config.model_regularizer,
            name="mod_rnn",
        )
        mod_mu_layer = tf.keras.layers.Dense(config.n_modes, name="mod_mu")
        mod_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="mod_sigma"
        )
        kl_div_layer = KLDivergenceLayer(
            config.theta_std_epsilon, config.loss_calc, name="kl_div"
        )
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        # ---------- Forward pass ---------- #

        # Encoder
        data = tf.keras.layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )
        data_drop = data_drop_layer(data)
        inf_rnn = inf_rnn_layer(data_drop)
        inf_mu = inf_mu_layer(inf_rnn)
        inf_sigma = inf_sigma_layer(inf_rnn)
        theta = theta_layer([inf_mu, inf_sigma])
        alpha = alpha_layer(theta)

        # Observation model
        mu = means_layer(data)
        D = covs_layer(data)
        m = mix_means_layer([alpha, mu])
        C = mix_covs_layer([alpha, D])
        ll_loss = ll_loss_layer([data, m, C])

        # Temporal prior
        theta_drop = theta_drop_layer(theta)
        mod_rnn = mod_rnn_layer(theta_drop)
        mod_mu = mod_mu_layer(mod_rnn)
        mod_sigma = mod_sigma_layer(mod_rnn)
        kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
        kl_loss = kl_loss_layer(kl_div)

        # ---------- Create model ---------- #
        inputs = {"data": data}
        outputs = {"ll_loss": ll_loss, "kl_loss": kl_loss, "theta": theta}
        name = config.model_name
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def get_amplitude(self):
        """Get the amplitude of the oscillators.

        Returns
        -------
        amplitude : np.ndarray
            The amplitude of the oscillators.
        """
        covs = self.model.get_layer("covs")
        return covs.amplitude(tf.constant(1)).numpy()

    def get_frequency(self):
        """Get the frequencies of the oscillators.

        Returns
        -------
        frequency : np.ndarray
            The frequencies of the oscillators.
        """
        covs = self.model.get_layer("covs")
        return covs.frequency(tf.constant(1)).numpy()

    def get_variance(self):
        """Get the variances of the oscillators.

        Returns
        -------
        variance : np.ndarray
            The variance of the oscillators.
        """
        covs = self.model.get_layer("covs")
        return tf.nn.softplus(covs.variance(tf.constant(1)).numpy())

    def get_oscillator_parameters(self):
        """Get the parameters of the oscillators.

        Returns
        -------
        oscillator_parameters : dict[str, np.ndarray]
            The parameters of the model. Keys are :code:`'amplitude'`,
            :code:`'frequency'` and :code:`'variance'`.
        """
        return {
            "amplitude": self.get_amplitude(),
            "frequency": self.get_frequency(),
            "variance": self.get_variance(),
        }
