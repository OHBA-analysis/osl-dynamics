"""Single-channel Dynamic Network Modes (SC-DyNeMo).

Note
----
This model is a work in progress. It has not been fully validated.
"""

from dataclasses import dataclass

from osl_dynamics.inference.layers import DampedOscillatorCovarianceMatricesLayer
from osl_dynamics.models.dynemo import Config as DynemoConfig
from osl_dynamics.models.dynemo import Model as DynemoModel
from osl_dynamics.utils import errors


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
            _msg = "Both sampling_frequency and learn_oscillator_amplitude must be specified."
            raise ValueError(_msg)

        if self.sampling_frequency <= 0:
            raise errors.SamplingFrequencyError(self.sampling_frequency)

        nyquist_frequency = self.sampling_frequency / 2
        if (
            self.oscillator_frequency_limit[0] < 0
            or self.oscillator_frequency_limit[1] > nyquist_frequency
        ):
            raise errors.FrequencyLimitError(
                self.oscillator_frequency_limit, self.sampling_frequency
            )
        if self.oscillator_damping_limit <= 0:
            raise errors.DampingLimitError(self.oscillator_damping_limit)


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

    def _select_covariance_layer(self):
        """Set the covariance layer to a DampedOscillatorCovarianceMatricesLayer.

        Returns
        -------
        covariance_layer : DampedOscillatorCovarianceMatricesLayer
            The covariance layer.
        """
        return DampedOscillatorCovarianceMatricesLayer(
            n=self.config.n_modes,
            m=self.config.n_channels,
            sampling_frequency=self.config.sampling_frequency,
            damping_limit=self.config.oscillator_damping_limit,
            frequency_limit=self.config.oscillator_frequency_limit,
            learn_amplitude=self.config.learn_oscillator_amplitude,
            learn=self.config.learn_covariances,
            name="covs",
        )

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
