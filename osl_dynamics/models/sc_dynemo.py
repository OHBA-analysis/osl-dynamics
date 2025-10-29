"""Single-channel Dynamic Network Modes (SC-DyNeMo)."""

from dataclasses import dataclass

from osl_dynamics.inference.layers import (
    DampedOscillatorCovarianceMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    NormalizationLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
)
from osl_dynamics.models.dynemo import Config as DynemoConfig
from osl_dynamics.models.dynemo import Model as DynemoModel


@dataclass
class Config(DynemoConfig):
    """Additional parameters for SC-DyNeMo.

    Parameters
    ----------
    sampling_frequency : float
        The sampling frequency of the data (Hz).
    oscillator_damping_limit : float, optional
        The damping limit (Hz). If None, it is set to 40.
    oscillator_frequency_limit : tuple[float, float], optional
        The frequency limit (Hz). If None, it is set to (0.1, sampling_frequency / 2).
    learn_oscillator_amplitude : bool
        Whether to learn the amplitude. If False, the amplitude is fixed to 1.
    """

    sampling_frequency: float = None
    oscillator_damping_limit: float = 40.0
    oscillator_frequency_limit: tuple[float, float] | None = None
    learn_oscillator_amplitude: bool = None

    def __post_init__(self):
        super().__post_init__()
        if (
            self.oscillator_frequency_limit is None
            and self.sampling_frequency is not None
        ):
            self.oscillator_frequency_limit = (0.1, self.sampling_frequency / 2)
        self.validate_sc_dynemo_params()

    def validate_sc_dynemo_params(self):
        """Validate the parameters of the model.

        Raises
        ------
        SamplingFrequencyError
            If the sampling frequency is not positive.
        FrequencyLimitError
            If the frequency limit is not between 0 and the Nyquist frequency.
        DampingLimitError
            If the damping limit is not positive.
        ValueError
            If any of the parameters are None.
        """
        non_optional_params = [
            self.sampling_frequency,
            self.learn_oscillator_amplitude,
        ]

        if any(param is None for param in non_optional_params):
            raise ValueError(
                "Both sampling_frequency and learn_oscillator_amplitude must be specified."
            )

        if self.sampling_frequency <= 0:
            raise ValueError(
                f"sampling_frequency must be greater than zero, got {self.sampling_frequency}"
            )

        nyquist_frequency = self.sampling_frequency / 2
        if self.oscillator_frequency_limit[1] > nyquist_frequency:
            raise ValueError(
                f"sampling_frequency needs to be less than {nyquist_frequency}, got {self.sampling_frequency}"
            )


class Model(DynemoModel):
    r"""Single-channel Dynamic Network Modes (SC-DyNeMo).

    This model is a single-channel version of DyNeMo.
    It should only be used for single-channel data which has been time-embedded.

    This model parameterises the covariance matrices as a set of damped oscillators.
    It should be used for single-channel data which has been time-embedded.
    The parameters are the damping, frequency and amplitude of the oscillators.
    They define the auto-covariance functions of the modes using the equation:

    .. math::
        R_j (\tau) = A_j e^{- \lambda_j \tau}\cos(2 \pi f_j \tau)

    where :math:`\lambda_j` is the damping, :math:`f_j` is the frequency and
    :math:`A_j` is the amplitude of the :math:`j`-th oscillator.
    :math:`\tau` is the time lag.

    The covariance matrices are then defined as a symmetric Toeplitz matrix
    with the auto-covariance function as the first row:

    .. math::
        \Sigma_j = \begin{bmatrix}
            R_j(0) & R_j(1) & \dots & R_j(p-1) \\
            R_j(1) & R_j(0) & \dots & R_j(p-2) \\
            \vdots & \vdots & \ddots & \vdots \\
            R_j(p-1) & R_j(p-2) & \dots & R_j(0)
        \end{bmatrix}

    where :math:`p` is the embedding dimension.

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
        covs_layer = DampedOscillatorCovarianceMatricesLayer(
            n=self.config.n_modes,
            m=self.config.n_channels,
            sampling_frequency=self.config.sampling_frequency,
            damping_limit=self.config.oscillator_damping_limit,
            frequency_limit=self.config.oscillator_frequency_limit,
            learn_amplitude=self.config.learn_oscillator_amplitude,
            learn=self.config.learn_covariances,
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

    def _oscillator_layer(self):
        """Get the covariance layer.

        Returns
        -------
        covariance_layer : DampedOscillatorMatricesLayer
            The covariance layer.
        """
        return self.model.get_layer("covs").oscillator_layer

    def get_frequency(self):
        """Get the frequencies of the oscillators.

        Returns
        -------
        oscillator_frequencies : np.ndarray
            The frequencies of the oscillators.
        """
        covs = self._oscillator_layer()
        return covs.frequency(1).numpy()

    def get_damping(self):
        """Get the damping of the oscillators.

        Returns
        -------
        oscillator_damping_factors : np.ndarray
            The damping of the oscillators.
        """
        covs = self._oscillator_layer()
        return covs.damping(1).numpy()

    def get_amplitude(self):
        """Get the amplitude of the oscillators.

        Returns
        -------
        oscillator_amplitudes : np.ndarray
            The amplitude of the oscillators.
        """
        covs = self._oscillator_layer()
        return covs.amplitude(1).numpy()

    def get_oscillator_parameters(self):
        """Get the parameters of the oscillators.

        Returns
        -------
        oscillator_parameters : dict[str, np.ndarray]
            The parameters of the model.
            Keys are "frequency", "damping" and "amplitude".
        """
        return {
            "frequency": self.get_frequency(),
            "damping": self.get_damping(),
            "amplitude": self.get_amplitude(),
        }
