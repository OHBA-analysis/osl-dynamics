"""Simulates a Gaussian Mixture Model and infers its parameters.

- This method is often used to select connections in coherence maps
  that have an abnormally high value.
"""

import numpy as np
from dynemo.analysis.gmm import fit_gmm

# Simulate a Gaussian mixture with two components
n_samples = 2560

component = np.random.binomial(1, 0.5, size=n_samples)

n1 = np.sum(component)
n2 = n_samples - n1

X = np.empty(n_samples)
X[component == 1] = np.random.normal(0, 0.5, size=n1)
X[component == 0] = np.random.normal(2, 1, size=n2)

# Fit a Gaussian mixture model
y = fit_gmm(X, n_fits=10, plot_filename="gmm.png")

print("Accuracy:", np.count_nonzero(y == component) / n_samples)
