import logging

import numpy as np
from taser.array_ops import get_one_hot
from taser.simulation import Simulation
from taser.utils.decorators import auto_repr, auto_yaml


class HiddenSemiMarkovSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        e_std: float = 0.2,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        stay_prob: float = 0.95,
        gamma_shape: float = 5,
        gamma_scale: float = 10,
    ):
        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob
        self.cumsum_off_diagonal_trans_prob = None
        self.stay_prob = stay_prob

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
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
            np.fill_diagonal(self.off_diagonal_trans_prob, 0)

            self.off_diagonal_trans_prob = (
                self.off_diagonal_trans_prob
                / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
            )

        if self.off_diagonal_trans_prob is not None:
            self.off_diagonal_trans_prob = self.off_diagonal_trans_prob

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        with np.printoptions(linewidth=np.nan):
            logging.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        self.cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob, axis=1
        )
        alpha_sim = np.zeros(self.n_samples, dtype=np.int)

        gamma_sample = np.random.default_rng().gamma
        random_sample = np.random.default_rng().uniform
        current_state = np.random.default_rng().integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha_sim):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)
            alpha_sim[
                current_position : current_position + state_lifetime
            ] = current_state

            tmp = random_sample()
            for kk in range(self.cumsum_off_diagonal_trans_prob.shape[1]):
                if tmp < self.cumsum_off_diagonal_trans_prob[current_state, kk]:
                    break

            current_position += state_lifetime
            current_state = kk

        logging.debug(f"n_states present in alpha sim = {len(np.unique(alpha_sim))}")

        one_hot_alpha_sim = get_one_hot(alpha_sim, n_states=self.n_states)

        logging.debug(f"one_hot_alpha_sim.shape = {one_hot_alpha_sim.shape}")

        return one_hot_alpha_sim
