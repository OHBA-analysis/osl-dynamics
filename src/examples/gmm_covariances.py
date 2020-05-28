import taser.plotting
from taser.inference import gmm
from taser.simulation import Simulation

simulation = Simulation()
data = simulation.time_series.T
simulation.plot_data()

covariances, means = gmm.learn_mu_sigma(
    data,
    n_states=simulation.n_states,
    n_channels=simulation.n_channels,
    learn_means=True,
)
cholesky_djs = gmm.find_cholesky_decompositions(covariances, means, learn_means=True)

taser.plotting.plot_covariances(cholesky_djs, fig_kwargs={"figsize": (20, 10)})
