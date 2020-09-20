"""Classes for simulating Hidden Semi-Markov Models (HSMMs).

"""

import logging

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml

_logger = logging.getLogger("VRAD")


class HSMMSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        n_states: int,
        sim_varying_means: bool,
        observation_error: float,
        covariances: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
        n_channels: int = None,
        random_covariance_weights: bool = False,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        random_seed: int = None,
        simulate: bool = True,
    ):
        if covariances is not None:
            n_channels = covariances.shape[1]

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob
        self.cumsum_off_diagonal_trans_prob = None

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            covariances=covariances,
            random_seed=random_seed,
            simulate=simulate,
        )

    def construct_off_diagonal_trans_prob(self):
        if self.off_diagonal_trans_prob is not None and (
            self.full_trans_prob is not None
        ):
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob must be "
                "specified. "
            )

        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_states, self.n_states])

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        self.cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob, axis=1
        )
        alpha_sim = np.zeros(self.n_samples, dtype=np.int)

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_state = self._rng.integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha_sim):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)
            alpha_sim[
                current_position : current_position + state_lifetime
            ] = current_state

            rand = random_sample()
            current_state = np.argmin(
                self.cumsum_off_diagonal_trans_prob[current_state] < rand
            )
            current_position += state_lifetime

        _logger.debug(f"n_states present in alpha sim = {len(np.unique(alpha_sim))}")

        one_hot_alpha_sim = get_one_hot(alpha_sim, n_states=self.n_states)

        _logger.debug(f"one_hot_alpha_sim.shape = {one_hot_alpha_sim.shape}")

        return one_hot_alpha_sim
