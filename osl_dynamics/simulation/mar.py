"""Multivariate autoregressive (MAR) observation model.

"""

import numpy as np


class MAR:
    """Class that generates data from a multivariate autoregressive (MAR) model.

    A :math:`p`-order MAR model can be written as

    .. math::
        x_t = A_1 x_{t-1} + ... + A_p x_{t-p} + \epsilon_t

    where :math:`\epsilon_t \sim N(0, \Sigma)`. The MAR model is therefore
    parameterized by the MAR coefficients (:math:`A`) and covariance
    (:math:`\Sigma`).

    Parameters
    ----------
    coeffs : np.ndarray
        Array of MAR coefficients, :math:`A`. Shape must be
        (n_states, n_lags, n_channels, n_channels).
    covs : np.ndarray
        Covariance of the error :math:`\epsilon_t`. Shape must be
        (n_states, n_channels) or (n_states, n_channels, n_channels).

    Note
    ----
    This model is also known as VAR or MVAR.
    """

    def __init__(self, coeffs, covs):
        # Validation
        if coeffs.ndim != 4:
            raise ValueError(
                "coeffs must be a (n_states, n_lags, n_channels, n_channels) array."
            )

        if covs.ndim == 2:
            covs = np.array([np.diag(c) for c in covs])

        if coeffs.shape[0] != covs.shape[0]:
            raise ValueError("Different number of states in coeffs and covs passed.")

        if coeffs.shape[-1] != covs.shape[-1]:
            raise ValueError("Different number of channels in coeffs and covs passed.")

        # Model parameters
        self.coeffs = coeffs
        self.covs = covs
        self.order = coeffs.shape[1]

        # Number of states and channels
        self.n_states = coeffs.shape[0]
        self.n_channels = coeffs.shape[2]

    def simulate_data(self, state_time_course):
        """Simulate time series data.

        We simulate MAR data based on the hidden state at each time point.

        Parameters
        ----------
        state_time_course : np.ndarray
            State time course. Shape must be (n_samples, n_states).
            States must be mutually exclusive.

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        n_samples = state_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate the noise term first
        for i in range(self.n_states):
            time_points_active = state_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] = np.random.multivariate_normal(
                np.zeros(self.n_channels),
                self.covs[i],
                size=n_time_points_active,
            )

        # Generate the MAR process
        for t in range(n_samples):
            state = state_time_course[t].argmax()
            for lag in range(min(t, self.order)):
                data[t] += np.dot(self.coeffs[state, lag], data[t - lag - 1])

        return data.astype(np.float32)
