from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vrad.data import Data
from vrad.files import example
from vrad.inference.states import match_covariances
from vrad.simulation import HSMMSimulation  # TODO: Update this to HSMM
from vrad.utils import plotting

n_samples = 128000
observation_error = 0.2
gamma_shape = 10
gamma_scale = 5

n_states = 5

covs = np.load(example.path / "hmm_cov.npy")

# TODO: Arguments need to change with update
sim = HSMMSimulation(
    n_samples=n_samples,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    zero_means=True,
    covariances=covs,
    random_seed=123,
)

data = Data(sim)

prepared = data[0]

full_covariance = np.cov(prepared.T)
no_diag = plotting.mean_diagonal(full_covariance)

plt.matshow(no_diag)
plt.suptitle("Data covariance (diagonal removed)")
plt.show()


sample_covs = data.covariance_sample(
    segment_length=[50 * i for i in range(1, 5)],
    n_segments=100,
    n_clusters=5,
)

covs, sample_covs = match_covariances(covs, sample_covs)

plotting.plot_matrices(sample_covs, main_title="GT covariances")
plotting.plot_matrices(
    [plotting.mean_diagonal(cov) for cov in sample_covs],
    main_title="GT covariances (diagonal removed)",
)

plotting.plot_matrices(covs, main_title="Sampled covariances")
plotting.plot_matrices(
    [plotting.mean_diagonal(cov) for cov in covs],
    main_title="Sampled covariances (diagonal removed)",
)
