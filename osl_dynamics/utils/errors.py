"""Custom errors."""


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
