"""Multivariate autoregressive (MAR) observation model.

"""

import numpy as np


class MAR:
    """Class that generates data from a multivariate autoregressive (MAR) model.

    A p-order MAR model can be written as

        x_t = coeffs_1 x_{t-1} + ... + coeffs_p x_{t-p} + eps_t

    where eps_t ~ N(0, covs). The MAR model is therefore parameterized by the MAR
    coefficients, coeffs_i, and covsariance, covs.

    This model is also known as VAR or MVAR.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of MAR coefficients.
        Shape must be (n_modes, n_lags, n_channels, n_channels).
    covs : np.ndarray
        Covariance of eps_t. Shape must be (n_modes, n_channels) or
        (n_modes, n_channels, n_channels).
    random_seed: int
        Seed for the random number generator.
    """

    def __init__(
        self,
        coeffs,
        covs,
        random_seed=None,
    ):
        # Validation
        if coeffs.ndim != 4:
            raise ValueError(
                "coeffs must be a (n_modes, n_lags, n_channels, n_channels) array."
            )

        if covs.ndim == 2:
            covs = np.array([np.diag(c) for c in covs])

        if coeffs.shape[0] != covs.shape[0]:
            raise ValueError("Different number of modes in coeffs and covs passed.")

        if coeffs.shape[-1] != covs.shape[-1]:
            raise ValueError("Different number of channels in coeffs and covs passed.")

        # Model parameters
        self.coeffs = coeffs
        self.covs = covs
        self.order = coeffs.shape[1]

        # Number of modes and channels
        self.n_modes = coeffs.shape[0]
        self.n_channels = coeffs.shape[2]

        # Setup random number generator
        self._rng = np.random.default_rng(random_seed)

    def simulate_data(self, mode_time_course):
        # NOTE: We assume mutually exclusive modes when generating the data

        n_samples = mode_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate the noise term first
        for i in range(self.n_modes):
            time_points_active = mode_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] = self._rng.multivariate_normal(
                np.zeros(self.n_channels),
                self.covs[i],
                size=n_time_points_active,
            )

        # Generate the MAR process
        for t in range(n_samples):
            mode = mode_time_course[t].argmax()
            for lag in range(min(t, self.order)):
                data[t] += np.dot(self.coeffs[mode, lag], data[t - lag - 1])

        return data.astype(np.float32)
