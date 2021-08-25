"""Example script for calculating covariances for initialisation using random segements
of the training data.

"""

import numpy as np
from vrad.data import Data
from vrad.files import example
from vrad.inference.states import match_covariances
from vrad.simulation import HSMM_MVN
from vrad.utils import plotting

# Ground truth covariances
gt_covs = np.load(example.path / "hmm_cov.npy")

# Simulate an HSMM
sim = HSMM_MVN(
    n_samples=128000,
    gamma_shape=10,
    gamma_scale=5,
    means="zero",
    covariances=gt_covs,
    random_seed=123,
)
data = Data(sim.time_series)

# Plot the covariance of the entire time series
full_cov = np.cov(data.time_series(), rowvar=False)
plotting.plot_matrices(full_cov, filename="full_cov.png")

# Sample covariances
sam_covs = data.covariance_sample(
    segment_length=[50 * i for i in range(1, 5)],
    n_segments=100,
    n_clusters=5,
)
gt_covs, sam_covs = match_covariances(gt_covs, sam_covs)

# Plot the ground truth and sampled covariances
plotting.plot_matrices(
    gt_covs, main_title="Sampled covariances", filename="gt_covs.png"
)
plotting.plot_matrices(
    sam_covs, main_title="Ground truth covariances", filename="sam_covs.png"
)
