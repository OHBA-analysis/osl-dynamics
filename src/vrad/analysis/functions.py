"""Helper functions for analysis.

"""

import numpy as np


def fourier_transform(
    data: np.ndarray,
    sampling_frequency: float,
    nfft: int = None,
    args_range: list = None,
) -> np.ndarray:
    """Calculates a Fast Fourier Transform (FFT).
    
    Parameters
    ----------
    data : np.ndarray
        Data with shape (n_samples, n_channels) to FFT.
    sampling_frequency : float
        Frequency used to sample the data (Hz).
    nfft : int
        Number of points in the FFT
    args_range : list
        Minimum and maximum indices of the FFT to keep.

    Returns
    -------
    np.ndarray
        FFT data.

    """
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


def nextpow2(x: int) -> int:
    """Next power of 2.

    Parameters
    ----------
    x : int

    Returns
    -------
    int
        The smallest power of two that is greater than or equal to the absolute
        value of x.

    """
    res = np.ceil(np.log2(x))
    return res.astype("int")


def validate_array(
    array: np.ndarray,
    correct_dimensionality: int,
    allow_dimensions: list,
    error_message: str,
) -> np.ndarray:
    """Checks if an array has been passed correctly.

    In particular this function checks the dimensionality of the array is correct.

    Parameters
    ----------
    array : np.ndarray
        Array to be checked.
    correct_dimensionality : int
        The desired number of dimensions in the array.
    allow_dimensions : int
        The number of dimensions that is acceptable for the passed array to have.
    error_message : str
        Message to print if the array is not valid.

    Returns:
    np.ndarray
        Array with the correct dimensionality.

    """
    array = np.array(array)

    # Add dimensions to ensure array has the correct dimensionality
    for dimensionality in allow_dimensions:
        if array.ndim == dimensionality:
            for i in range(correct_dimensionality - dimensionality):
                array = array[np.newaxis, ...]

    # Check no other dimensionality has been passed
    if array.ndim != correct_dimensionality:
        raise ValueError(error_message)

    return array
