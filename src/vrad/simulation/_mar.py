"""Multivariate autoregressive (MAR) observation model.

"""

import numpy as np


class MAR:
    """Class that generates data from a multivariate autoregressive (MAR) model.

    This model is also known as VAR, MVAR.

    A p-order MAR model can be written as

        x_t = A_1 x_{t-1} + ... + A_p x_{t-p} + eps_t

    where eps_t ~ N(0, v). The MAR model is therefore parameterized by the MAR
    coefficients, A_i, and variance, v.

    Paramters
    ---------
    coeffs : np.ndarray
        Array of MAR coefficients. Shape must be (n_states, n_lags, n_channels,
        n_channels).
    var : np.ndarray
        Variance of eps_t. Shape must be (n_states, n_channels)
    random_seed: int
        Seed for the random number generator.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        var: float,
        random_seed: int = None,
    ):
        # Validation
        if coeffs.ndim != 4:
            raise ValueError("coeffs must be a 4D array.")

        if var.ndim != 2:
            raise ValueError("var must be a 2D array.")

        if coeffs.shape[0] != var.shape[0]:
            raise ValueError("Different number of states in coeffs and var passed.")

        if coeffs.shape[-1] != var.shape[-1]:
            raise ValueError("Different number of channels in coeffs and var passed.")

        # Model parameters
        self.A = coeffs
        self.v = var
        self.order = coeffs.shape[1]

        # Number of states and channels
        self.n_states = coeffs.shape[0]
        self.n_channels = coeffs.shape[2]

        # Setup random number generator
        self._rng = np.random.default_rng(random_seed)

    def simulate_data(self, state_time_course):
        # NOTE: We assume mutually exclusive states when generating the data

        n_samples = state_time_course.shape[0]

        # Noise term
        eps = np.empty([n_samples, self.n_channels])
        for i in range(self.n_states):
            time_points_active = state_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            eps[time_points_active] = self._rng.normal(
                np.zeros(self.n_channels),
                self.v[i],
                size=(n_time_points_active, self.n_channels),
            )

        # Generate data
        data = np.empty([n_samples, self.n_channels])
        data[: self.order] = eps[: self.order]
        for i in range(self.order, n_samples):
            state = state_time_course[i].argmax()
            data[i] = (
                np.sum(
                    [self.A[state, j] @ data[i - j] for j in range(self.order)], axis=0
                )
                + eps[i]
            )

        return data.astype(np.float32)
