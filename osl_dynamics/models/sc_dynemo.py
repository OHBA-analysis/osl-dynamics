"""Single-channel Dynamic Network Modes (SC-DyNeMo)."""

from osl_dynamics.inference.layers import DampedOscillatorMatricesLayer
from osl_dynamics.models.dynemo import Config as DynemoConfig
from osl_dynamics.models.dynemo import Model as DynemoModel


class SamplingFrequencyError(ValueError):
    """Raised when the sampling frequency is not positive."""

    def __init__(self, sampling_frequency):
        """Initialize a SamplingFrequencyError.

        Parameters
        ----------
        sampling_frequency : float
            The sampling frequency.
        """
        super().__init__(
            f"Sampling frequency must be positive, got {sampling_frequency}",
        )


class FrequencyLimitError(ValueError):
    """Raised when the frequency limit is not between 0 and the Nyquist frequency."""

    def __init__(self, frequency_limit, sampling_frequency):
        """Initialize a FrequencyLimitError.

        Parameters
        ----------
        frequency_limit : tuple[float, float]
            The frequency limit.
        sampling_frequency : float
            The sampling frequency.
        """
        super().__init__(
            f"Frequency limit must be between 0 "
            f"and the Nyquist frequency ({sampling_frequency / 2}). "
            f"Got {frequency_limit}.",
        )


class DampingLimitError(ValueError):
    """Raised when the damping limit is not positive."""

    def __init__(self, damping_limit):
        """Initialize a DampingLimitError.

        Parameters
        ----------
        damping_limit : float
            The damping limit.
        """
        super().__init__(
            f"Damping limit must be positive, got {damping_limit}",
        )


class Config(DynemoConfig):
    """Settings for SC-DyNeMo.

    Parameters
    ----------
    sampling_frequency : float
        The sampling frequency of the data (Hz).
    damping_limit : float, optional
        The damping limit (Hz). If None, it is set to 40.
    frequency_limit : tuple[float, float], optional
        The frequency limit (Hz). If None, it is set to (0.1, sampling_frequency / 2).
    learn_amplitude : bool
        Whether to learn the amplitude. If False, the amplitude is fixed to 1.
    n_channels: int
        The number of channels should be the number of embeddings.
        It is named n_channels to be consistent with the DyNeMo Config class.
    """

    sampling_frequency: float
    damping_limit: float = 40.0
    frequency_limit: tuple[float, float] | None = None
    learn_amplitude: bool

    def __post_init__(self):
        super().__post_init__()
        if self.frequency_limit is None:
            self.frequency_limit = (0.1, self.sampling_frequency / 2)
        self.validate_sc_dynemo_params()

    def validate_sc_dynemo_params(self):
        if self.sampling_frequency <= 0:
            raise SamplingFrequencyError(self.sampling_frequency)

        nyquist_frequency = self.sampling_frequency / 2
        if self.frequency_limit[0] < 0 or self.frequency_limit[1] > nyquist_frequency:
            raise FrequencyLimitError(self.frequency_limit, self.sampling_frequency)
        if self.damping_limit <= 0:
            raise DampingLimitError(self.damping_limit)


class Model(DynemoModel):
    r"""Single-channel Dynamic Network Modes (SC-DyNeMo).

    This model is a single-channel version of DyNeMo.
    It should only be used for single-channel data which has been time-embedded.

    This model parameterises the covariance matrices as a set of damped oscillators.
    It should be used for single-channel data which has been time-embedded.
    The parameters are the damping, frequency and amplitude of the oscillators.
    They define the auto-covariance functions of the modes using the equation:

    .. math::
        R_j (\tau) = A_j e^(- \lambda_j \tau)\cos(2 \pi f_j \tau)

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

    def _select_covariance_layer(self):
        """Set the covariance layer to a DampedOscillatorMatricesLayer.

        Returns
        -------
        DampedOscillatorMatricesLayer
            The covariance layer.
        """
        return DampedOscillatorMatricesLayer(
            n=self.config.n_modes,
            m=self.config.n_channels,
            sampling_frequency=self.config.sampling_frequency,
            damping_limit=self.config.damping_limit,
            frequency_limit=self.config.frequency_limit,
            learn_amplitude=self.config.learn_amplitude,
            learn=self.config.learn_covariances,
            name="covs",
        )
