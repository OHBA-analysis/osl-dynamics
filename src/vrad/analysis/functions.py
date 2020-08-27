"""Helper functions for analysis.

"""

import numpy as np
from scipy.optimize import curve_fit


def residuals_gaussian_fit(data):
    """Fits a Gaussian to data and returns the residuals squared."""
    residuals_squared = 0
    n_components, n_f = data.shape
    x = range(n_f)
    for i in range(n_components):
        y = data[i, :] / max(data[i, :])
        params, _ = curve_fit(gaussian, x, y)
        fit = gaussian(x, params[0], params[1], params[2])
        residuals_squared += sum((y - fit) ** 2)
    return residuals_squared


def fourier_transform(data, sampling_frequency, nfft=None, args_range=None):
    """Calculates a Fast Fourier Transform (FFT)."""

    # Number of data points
    n = data.shape[-1]

    # Number of FFT data points to calculate
    if nfft is None:
        nfft = max(256, 2 ** nextpow2(n))

    # Calculate the FFT
    X = np.fft.fft(data, nfft) / sampling_frequency

    # Only keep the desired frequency range
    if args_range is not None:
        X = X[:, :, args_range[0] : args_range[1]]

    return X


def gaussian(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def nextpow2(x):
    """Returns the smallest power of two that is greater than or equal to the
    absolute value of x.
    """
    res = np.ceil(np.log2(x))
    return res.astype("int")
