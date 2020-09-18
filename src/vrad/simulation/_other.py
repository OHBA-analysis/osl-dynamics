"""Classes for other simulations.

"""

from vrad.simulation import Simulation


class VRADSimulation(Simulation):
    """Simulate a dataset from the covariances and state time course of a model."""

    def __init__(self, covariances, state_time_course):
        super().__init__(
            n_samples=state_time_course.shape[0],
            n_channels=covariances.shape[-1],
            n_states=covariances.shape[0],
            sim_varying_means=False,
            covariances=covariances,
            observation_error=0.0,
            random_covariance_weights=False,
            simulate=False,
        )
        self.state_time_course = state_time_course

    def generate_states(self):
        return self.state_time_course
