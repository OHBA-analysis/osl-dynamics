import matplotlib.pyplot as plt
import numpy as np


def get_one_hot(values, n_states):
    """

    Parameters
    ----------
    values
    n_states

    Returns
    -------

    """
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_states])


class Simulation:
    """

    """

    simulation_options = ["sequence_hmm", "uni_hmm", "hmm", "random"]

    def __init__(
        self,
        sim_type: str = "sequence_hmm",
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        markov_lag: int = 1,
        stay_prob: float = 0.95,
        random_covariance_weights: bool = False,
        e_std: float = 0.2,
    ):

        simulation_options = {
            "sequence_hmm": self.hmm,
            "uni_hmm": self.uni_hmm,
            "hmm": self.hmm,
            "random": self.random_hmm,
        }

        if sim_type not in simulation_options.keys():
            raise ValueError(
                f"The simulation types available are {', '.join(simulation_options.keys())}"
            )

        self.sim_type = sim_type
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_states = n_states
        self.sim_varying_means = sim_varying_means
        self.markov_lag = markov_lag
        self.stay_prob = stay_prob
        self.random_covariance_weights = random_covariance_weights
        self.e_std = e_std

        self.alpha_sim = simulation_options[self.sim_type]()
        self.djs = self.create_djs()
        self.data_sim = self.simulate_data()

    def hmm(self) -> np.ndarray:
        """

        Returns
        -------

        """
        alpha_sim = np.zeros((self.n_samples, self.n_states))
        z = np.zeros([self.n_samples + 1], dtype=int)

        single_trans_prob = (1 - self.stay_prob) / self.n_states
        hmm_trans_prob = np.ones((self.n_states, self.n_states)) * single_trans_prob
        hmm_trans_prob[np.diag_indices(self.n_states)] = self.stay_prob

        hmm_cumsum_trans_prob = np.cumsum(hmm_trans_prob, axis=1)

        rands = np.random.rand(self.n_samples)

        for tt in range(self.markov_lag - 1, self.n_samples):
            tmp = rands[tt]
            for kk in range(hmm_cumsum_trans_prob.shape[1]):

                if tmp < hmm_cumsum_trans_prob[z[tt - self.markov_lag], kk]:
                    z[tt] = kk
                    break
            alpha_sim[tt, z[tt]] = 1

        return alpha_sim

    def random_hmm(self) -> np.ndarray:
        """

        Returns
        -------

        """
        # TODO: Ask Mark what FOS is short for (frequency of state?)
        z = np.random.choice(self.n_states, size=self.n_samples)
        alpha_sim = get_one_hot(z, self.n_states)
        return alpha_sim

    def uni_hmm(self) -> np.ndarray:
        """

        Returns
        -------

        """
        hmm_trans_prob = np.ones([2, 2]) * (1 - self.stay_prob)
        hmm_trans_prob[np.diag_indices(2)] = self.stay_prob
        hmm_cumsum_trans_prob = np.cumsum(hmm_trans_prob, axis=1)

        z = np.zeros([self.n_samples, self.n_states], int)
        alpha_sim = np.zeros((self.n_samples, self.n_states))
        rands = np.random.rand(self.n_samples, self.n_states)

        for tt in range(self.markov_lag - 1, self.n_samples):
            for kk in range(self.n_states):

                tmp = rands[tt, kk]
                for jj in range(hmm_cumsum_trans_prob.shape[1]):
                    if tmp < hmm_cumsum_trans_prob[z[tt - self.markov_lag, kk], jj]:
                        z[tt, kk] = jj
                        break
                alpha_sim[tt, kk] = z[tt, kk]

        return alpha_sim

    def plot_alphas(self, n_points: int = 1000):
        """

        Parameters
        ----------
        n_points

        Returns
        -------

        """
        plt.figure()
        plt.plot(self.alpha_sim[0:n_points])
        plt.show()
        return

    def create_djs(self, identity_factor: float = 0.0001) -> np.ndarray:
        """

        Parameters
        ----------
        identity_factor

        Returns
        -------

        """
        if self.random_covariance_weights:
            tilde_cov_weights = np.random.normal(
                size=(self.n_states, self.n_channels, self.n_channels)
            )
        else:
            tilde_cov_weights = np.zeros(
                (self.n_states, self.n_channels, self.n_channels)
            )
            np.fill_diagonal(
                tilde_cov_weights[: self.n_states, : self.n_states, : self.n_states],
                val=1,
            )

        scaled_identity = (
            np.tile(np.eye(self.n_channels), [self.n_states, 1, 1]) * identity_factor
        )
        djs_sim = (
            tilde_cov_weights @ tilde_cov_weights.transpose([0, 2, 1]) + scaled_identity
        )

        normalisation = np.trace(djs_sim, axis1=1, axis2=2).reshape((-1, 1, 1))
        djs_sim /= normalisation
        return djs_sim

    def simulate_data(self) -> np.ndarray:
        """

        Returns
        -------

        """
        if self.sim_varying_means:
            mus_sim = np.random.normal((self.n_states, self.n_channels))
        else:
            mus_sim = np.zeros((self.n_states, self.n_channels))

        mus = np.sum(self.alpha_sim.reshape((-1, self.n_states, 1)) * mus_sim, axis=1)
        cs = np.sum(self.alpha_sim[:, :, np.newaxis, np.newaxis] * self.djs, axis=1)

        signal = np.zeros((self.n_channels, self.n_samples))
        for tt in range(self.n_samples):
            signal[:, tt] = np.random.multivariate_normal(mus[tt, :], cs[tt, :, :])

        noise = np.random.normal(
            loc=0, scale=self.e_std, size=(self.n_channels, self.n_samples)
        )
        data_sim = signal + noise

        return data_sim.astype(np.float32)

    def plot_data(self, n_points: int = 1000):
        """

        Parameters
        ----------
        n_points
        """
        n_points = min(n_points, self.n_samples)
        fig, y_axes = plt.subplots(
            1, min(self.n_channels, 10), figsize=(20, 3), sharey="row"
        )
        for y_axis, y_channel in zip(y_axes, self.data_sim):
            y_axis.plot(np.arange(n_points), y_channel[:n_points])
        fig, alpha_axes = plt.subplots(1, min(self.n_states, 10), figsize=(15, 3))
        for alpha_axis, alpha_channel in zip(alpha_axes, self.alpha_sim.T):
            alpha_axis.plot(np.arange(n_points), alpha_channel[:n_points])
        plt.tight_layout()
        plt.show()
