from unittest import TestCase
from taser.simulation import Simulation
import numpy as np


class SimulationTest(TestCase):
    def setUp(self) -> None:
        """

        """
        np.random.seed(0)
        self.simulation = Simulation(random_covariance_weights=True)

    def test_create_djs(self):
        """


        """
        s = self.simulation

        np.random.seed(0)
        djs = np.zeros((s.n_states, s.n_channels, s.n_channels))
        lamb = 0.0001

        tilde_cov_weights = np.random.normal(
            size=[s.n_states, s.n_channels, s.n_channels]
        )

        for i in range(s.n_states):
            djs[i, :, :] = (
                np.matmul(
                    tilde_cov_weights[i, :, :], np.transpose(tilde_cov_weights[i, :, :])
                )
                + np.eye(s.n_channels) * lamb
            )

        normalisation = np.trace(djs, axis1=1, axis2=2).reshape((-1, 1, 1))
        djs /= normalisation

        np.random.seed(0)
        self.assertTrue(
            np.all(s.create_djs() == djs), "NumPy methods not producing same results"
        )
