"""Functions to manipulate and analyse inferred mode/state data.

"""

import itertools
import logging

import numpy as np
from scipy import signal

from osl_dynamics import array_ops, inference

_logger = logging.getLogger("osl-dynamics")


def autocorr_from_tde_cov(
    covs, n_embeddings, pca_components=None, sampling_frequency=None
):
    """Auto/cross-correlation function from the mode covariance matrices.

    Parameters
    ----------
    covs : np.ndarray
        Covariance matrix of time-delay embedded data. Shape must be
        (n_channels, n_channels) or (n_modes, n_channels, n_channels).
    n_embeddings : int
        Number of embeddings.
    pca_components : np.ndarray, optional
        PCA components used for dimensionality reduction. Only needs to be
        passed if PCA was performed on the time embedded data.
    sampling_frequency : float, optional
        Sampling_frequency in Hz.

    Returns
    -------
    tau : np.ndarray
        Time lags in samples if `sampling_frequency=None`, otherwise in seconds.
        Shape is (n_lags).
    acfs : np.ndarray
        Auto/cross-correlation functions. Shape is (n_channels, n_channels,
        n_lags) or (n_modes, n_channels, n_channels, n_lags).
    """
    # Validation
    error_message = (
        "covs must be of shape (n_channels, n_channels) or "
        + "(n_modes, n_channels, n_channels) or "
        + "(n_subjects, n_modes, n_channels, n_channels)."
    )
    covs = array_ops.validate(
        covs,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    if sampling_frequency is None:
        sampling_frequency = 1

    # Get covariance of time embedded data
    if pca_components is not None:
        te_covs = reverse_pca(covs, pca_components)
    else:
        te_covs = covs

    # Dimensions
    n_subjects = te_covs.shape[0]
    n_modes = te_covs.shape[1]
    n_parcels = te_covs.shape[-1] // n_embeddings
    n_lags = 2 * n_embeddings - 1

    # Take mean of elements from the time embedded covariances that
    # correspond to the auto/cross-correlation function
    blocks = te_covs.reshape(
        n_subjects,
        n_modes,
        n_parcels,
        n_embeddings,
        n_parcels,
        n_embeddings,
    )
    acfs = np.empty([n_subjects, n_modes, n_parcels, n_parcels, n_lags])
    for i in range(n_lags):
        acfs[:, :, :, :, i] = np.mean(
            np.diagonal(blocks, offset=i - n_embeddings + 1, axis1=3, axis2=5),
            axis=-1,
        )

    # Time lags axis
    tau = np.arange(-(n_embeddings - 1), n_embeddings) / sampling_frequency

    return tau, np.squeeze(acfs)


def raw_covariances(
    mode_covariances,
    n_embeddings,
    pca_components,
    zero_lag=False,
):
    """Covariance matrix of the raw channels.

    PCA and time embedding is reversed to give you to the covariance matrix
    of the raw channels.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariance matrices.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.
    zero_lag : bool, optional
        Should we return just the zero-lag elements? Otherwise, we return
        the mean over time lags.

    Returns
    -------
    raw_covs : np.ndarray
        Covariance matrix for raw channels.
    """
    # Validation
    error_message = (
        "mode_covariances must be of shape (n_channels, n_channels) or "
        + "(n_modes, n_channels, n_channels) or "
        + "(n_subjects, n_modes, n_channels, n_channels)."
    )
    mode_covariances = array_ops.validate(
        mode_covariances,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    # Get covariance of time embedded data
    te_covs = reverse_pca(mode_covariances, pca_components)

    if zero_lag:
        # Return the zero-lag elements only
        raw_covs = te_covs[
            :,
            :,
            n_embeddings // 2 :: n_embeddings,
            n_embeddings // 2 :: n_embeddings,
        ]

    else:
        # Return block means
        n_subjects = te_covs.shape[0]
        n_modes = te_covs.shape[1]
        n_parcels = te_covs.shape[-1] // n_embeddings

        n_parcels = te_covs.shape[-1] // n_embeddings
        blocks = te_covs.reshape(
            n_subjects,
            n_modes,
            n_parcels,
            n_embeddings,
            n_parcels,
            n_embeddings,
        )
        block_diagonal = blocks.diagonal(0, 2, 4)
        diagonal_means = block_diagonal.diagonal(0, 2, 3).mean(3)

        raw_covs = blocks.mean((3, 5))
        raw_covs[:, :, np.arange(n_parcels), np.arange(n_parcels)] = diagonal_means

    return np.squeeze(raw_covs)


def reverse_pca(covariances, pca_components):
    """Reverses the effect of PCA on covariance matrices.

    Parameters
    ----------
    covariances : np.ndarray
        Covariance matrices.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.

    Returns
    -------
    covariances : np.ndarray
        Covariance matrix of the time embedded data.
    """
    if covariances.shape[-1] != pca_components.shape[-1]:
        raise ValueError(
            "Covariance matrix and PCA components have incompatible shapes: "
            + f"covariances.shape={covariances.shape}, "
            + f"pca_components.shape={pca_components.shape}."
        )

    return pca_components @ covariances @ pca_components.T


def state_activations(state_time_course):
    """Calculate state activations from a state time course.

    Given a state time course (strictly binary), calculate the beginning and
    end of each activation of each state. Accepts a 1D or 2D array. If a 1D
    array is passed, it is assumed to be a single state time course.

    Either an array of ints or an array of :code:`bool` is accepted, but if
    :code:`int` are passed they should be explicitly 0 or 1.

    Parameters
    ----------
    state_time_course : numpy.ndarray or list of numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    slices: list of list of slice
        List containing state activations (index) in the order they occur for
        each state. This cannot necessarily be converted into an array as an
        equal number of elements in each array is not guaranteed.
    """
    # Make sure we have a list of numpy arrays
    shape_error_message = (
        "State time course must be a 1D, 2D or 3D array or list of 2D arrays."
    )
    if isinstance(state_time_course, np.ndarray):
        if state_time_course.ndim == 3 or state_time_course.dtype == object:
            state_time_course = list(state_time_course)
        if state_time_course.ndim == 2:
            state_time_course = [state_time_course]
        elif state_time_course.ndim == 1:
            state_time_course = [state_time_course[:, np.newaxis]]
        else:
            raise ValueError(shape_error_message)
    elif isinstance(state_time_course, list):
        if not all(isinstance(stc, np.ndarray) for stc in state_time_course):
            raise ValueError(shape_error_message)
        if not all(stc.ndim == 2 for stc in state_time_course):
            raise ValueError(shape_error_message)

    # Make sure the list of arrays is of type bool
    type_error_message = (
        "State time course must be strictly binary. "
        "This can either be np.bools or np.ints with values 0 and 1."
    )
    bool_state_time_course = []
    for stc in state_time_course:
        if np.issubdtype(stc.dtype, np.integer):
            if np.all(np.isin(stc, [0, 1])):
                bool_state_time_course.append(stc.astype(bool))
        elif np.issubdtype(stc.dtype, np.bool_):
            bool_state_time_course.append(stc)
        else:
            raise ValueError(type_error_message)

    # Get the slices where each state is True
    slices = [
        [array_ops.ezclump(column) for column in stc.T]
        for stc in bool_state_time_course
    ]
    return slices


def lifetimes(state_time_course, sampling_frequency=None, squeeze=True):
    """Calculate state lifetimes from a state time course.

    Given a state time course (one-hot encoded), calculate the lifetime of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, or n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.
    squeeze : bool, optional
        If :code:`True`, squeeze the output to remove singleton dimensions.

    Returns
    -------
    lts : list of numpy.ndarray
        List containing an array of lifetimes in the order they occur for each
        state. This cannot necessarily be converted into an array as an equal
        number of elements in each array is not guaranteed. Shape is
        (n_subjects, n_states, n_activations) or (n_states, n_activations).
    """
    sampling_frequency = sampling_frequency or 1
    slices = state_activations(state_time_course)

    result = [
        [
            np.array([array_ops.slice_length(slice_) for slice_ in state_slices])
            / sampling_frequency
            for state_slices in subject_slices
        ]
        for subject_slices in slices
    ]

    if not squeeze:
        return result

    if len(result) == 1:
        result = result[0]
        if len(result) == 1:
            result = result[0]

    return result


def lifetime_statistics(state_time_course, sampling_frequency=None):
    """Calculate statistics of the lifetime distribution of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    means : np.ndarray
        Mean lifetime of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    std : np.ndarray
        Standard deviation of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    lifetimes_ = lifetimes(
        state_time_course,
        sampling_frequency=sampling_frequency,
        squeeze=False,
    )
    means = np.squeeze(array_ops.list_means(lifetimes_))
    stds = np.squeeze(array_ops.list_stds(lifetimes_))
    return means, stds


def mean_lifetimes(state_time_course, sampling_frequency=None):
    """Calculate the mean lifetime of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    mlt : np.ndarray
        Mean lifetime of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    return lifetime_statistics(state_time_course, sampling_frequency)[0]


def intervals(state_time_course, sampling_frequency=None, squeeze=True):
    """Calculate state intervals from a state time course.

    An interval is the duration between successive visits for a particular
    state.

    Parameters
    ----------
    state_time_course : list or numpy.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the intervals in seconds.
    squeeze : bool, optional
        If :code:`True`, squeeze the output to remove singleton dimensions.

    Returns
    -------
    intvs : list of numpy.ndarray
        List containing an array of intervals in the order they occur for each
        state. This cannot necessarily be converted into an array as an equal
        number of elements in each array is not guaranteed. Shape is
        (n_subjects, n_states, n_activations) or (n_states, n_activations).
    """
    sampling_frequency = sampling_frequency or 1
    slices = state_activations(state_time_course)
    result = []
    for subject_slice in slices:
        r = []
        for state_slices in subject_slice:
            a, b = itertools.tee(state_slices)
            next(b, None)
            state_slices_iter = zip(a, b)
            r.append(
                np.array(
                    [
                        slice_1.start - slice_0.stop
                        for slice_0, slice_1 in state_slices_iter
                    ]
                )
                / sampling_frequency
            )
        result.append(r)

    if not squeeze:
        return result

    if len(result) == 1:
        result = result[0]
        if len(result) == 1:
            result = result[0]

    return result


def interval_statistics(state_time_course, sampling_frequency=None):
    """Calculate statistics of the interval distribution of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    means : np.ndarray
        Mean interval of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    std : np.ndarray
        Standard deviation of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    intervals_ = intervals(
        state_time_course, sampling_frequency=sampling_frequency, squeeze=False
    )
    means = np.squeeze(array_ops.list_means(intervals_))
    stds = np.squeeze(array_ops.list_stds(intervals_))
    return means, stds


def mean_intervals(state_time_course, sampling_frequency=None):
    """Calculate the mean interval of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If passed returns the intervals in seconds.

    Returns
    -------
    mlt : np.ndarray
        Mean interval of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    return interval_statistics(state_time_course, sampling_frequency)[0]


def fractional_occupancies(state_time_course):
    """Calculate the fractional occupancy.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).

    Returns
    -------
    fo : np.ndarray
        The fractional occupancy of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    if isinstance(state_time_course, np.ndarray):
        if state_time_course.ndim == 2:
            state_time_course = [state_time_course]
        elif state_time_course.ndim != 3:
            raise ValueError(
                "A (n_subjects, n_samples, n_states) or "
                "(n_samples, n_states) array must be passed."
            )
    fo = [np.sum(stc, axis=0) / stc.shape[0] for stc in state_time_course]
    return np.squeeze(fo)


def switching_rates(state_time_course, sampling_frequency=None):
    """Calculate the switching rate.

    This is defined as the number of state activations per second.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, defaults to 1 Hz.

    Returns
    -------
    sr : np.ndarray
        The switching rate of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    if isinstance(state_time_course, np.ndarray):
        if state_time_course.ndim == 2:
            state_time_course = [state_time_course]
        elif state_time_course.ndim == 3:
            state_time_course = list(state_time_course)

    # Set sampling frequency
    sampling_frequency = sampling_frequency or 1

    # Loop through subjects
    sr = []
    for subject in state_time_course:
        n_samples, n_states = subject.shape

        # Number of activations for each state
        d = np.diff(subject, axis=0)
        counts = np.array([len(d[:, i][d[:, i] == 1]) for i in range(n_states)])

        # Calculate switching rates
        sr.append(counts * sampling_frequency / n_samples)

    return np.squeeze(sr)


def mean_amplitudes(state_time_course, data):
    """Calculate mean amplitude for bursts.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    data : list or np.ndarray
        Single channel time series data (before calculating the amplitude
        envelope). Shape must be (n_subjects, n_samples, 1) or (n_samples, 1).

    Returns
    -------
    amp : np.ndarray
        Mean amplitude of the data for each state.
        Shape is (n_subjects, n_states) or (n_states,).
    """
    if isinstance(state_time_course, np.ndarray):
        if state_time_course.ndim == 2:
            state_time_course = [state_time_course]
        elif state_time_course.ndim == 3:
            state_time_course = list(state_time_course)

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = [data]
        elif data.ndim == 3:
            data = list(data)

    n_subjects = len(state_time_course)
    n_states = state_time_course[0].shape[1]

    # Calculate amplitude envelope of data
    data = [abs(signal.hilbert(d, axis=0)) for d in data]

    # Calculate mean amplitude envelope when each state is on
    amp = np.empty([n_subjects, n_states])
    for i in range(n_subjects):
        for j in range(n_states):
            amp[i, j] = np.mean(data[i][state_time_course[i][:, j] == 1])

    return np.squeeze(amp)


def fano_factor(state_time_course, window_lengths, sampling_frequency=1.0):
    """Calculate the Fano factor.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    window_lengths : list or np.ndarray
        Window lengths to use. Must be in samples.
    sampling_frequency : float, optional
        Sampling frequency in Hz.

    Returns
    -------
    F : list of np.ndarray
        Fano factor. Shape is (n_subjects, n_window_lengths, n_states) or
        (n_window_lengths, n_states).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]

    # Loop through subjects
    F = []
    for subject in state_time_course:
        n_samples = subject.shape[0]
        n_states = subject.shape[1]
        F.append([])

        # Loop through window lengths
        for window_length in window_lengths:
            w = int(window_length * sampling_frequency)
            n_windows = n_samples // w
            tc = subject[: n_windows * w]
            tc = tc.reshape(n_windows, w, n_states)

            # Loop through windows
            counts = []
            for window in tc:
                # Number of activations
                d = np.diff(window, axis=0)
                c = []
                for i in range(n_states):
                    c.append(len(d[:, i][d[:, i] == 1]))
                counts.append(c)

            # Calculate Fano factor
            counts = np.array(counts)
            F[-1].append(np.std(counts, axis=0) ** 2 / np.mean(counts, axis=0))

    return np.squeeze(F)


def calc_trans_prob_matrix(state_time_course, n_states=None):
    """Calculate subject-specific transition probability matrices.

    Parameters
    ----------
    state_time_course : list of np.ndarray or np.ndarray
        State time courses. Shape must be (n_subjects, n_samples, n_states)
        or (n_samples, n_states).
    n_states : int, optional
        Number of states.

    Returns
    -------
    trans_prob : np.ndarray
        Subject-specific transition probability matrices. Shape is (n_subjects,
        n_states, n_states).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]
    trans_prob = []
    for stc in state_time_course:
        stc_argmax = stc.argmax(axis=1)
        vals, counts = np.unique(
            stc_argmax[np.arange(2)[None, :] + np.arange(len(stc_argmax) - 1)[:, None]],
            axis=0,
            return_counts=True,
        )
        if n_states is None:
            n_states = stc_argmax.max() + 1
        tp = np.zeros((n_states, n_states))
        tp[vals[:, 0], vals[:, 1]] = counts
        with np.errstate(divide="ignore", invalid="ignore"):
            tp /= tp.sum(axis=1)[:, None]
        trans_prob.append(np.nan_to_num(tp))
    return np.squeeze(trans_prob)


def simple_moving_average(data, window_length, step_size):
    """Calculate imple moving average.

    This function can be used to calculate a sliding window fractional occupancy
    from a state time course. This was done in `Baker et al. (2014)
    <https://elifesciences.org/articles/01867>`_.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    window_length : int
        Number of data points in a window.
    step_size : int
        Step size for shifting the window.

    Returns
    -------
    mov_avg : np.ndarray
        Mean for each window.
    """
    # Get number of samples and modes
    n_samples = data.shape[0]
    n_modes = data.shape[1]

    # Pad the data
    data = np.pad(data, window_length // 2)[
        :,
        window_length // 2 : window_length // 2 + n_modes,
    ]

    # Define indices of time points to calculate a moving average
    time_idx = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    # Preallocate an array to hold moving average values
    mov_avg = np.empty([n_windows, n_modes], dtype=np.float32)

    # Compute simple moving average
    for n in range(n_windows):
        j = time_idx[n]
        mov_window = data[j : j + window_length]
        mov_avg[n] = np.mean(mov_window, axis=0)

    return mov_avg


def partial_covariances(data, alpha):
    r"""Calculate partial covariances.

    Returns the multiple regression parameters estimates of the state/mode time
    courses regressed onto the data from each channel. The regression parameters
    are referred to as 'partial covariances'.

    We fit the regression:

    .. math::
        Y_i = X \beta_i + \epsilon

    where:

    - :math:`Y_i` is (n_samples, 1) the data amplitude/envelope/power/absolute
      time course at channel :math:`i`.
    - :math:`X` is (n_samples, n_states) matrix of the variance normalised
      state/mode time courses.
    - :math:`\beta_i` is an (n_states, 1) vector of multiple regression
      parameters for channel :math:`i`.
    - :math:`\epsilon` is the error.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        Training data for each subject. Shape is (n_subjects, n_samples,
        n_channels) or (n_samples, n_channels).
    alpha : np.ndarray or list of np.ndarray
        State/mode time courses for each subject. Shape is (n_subjects,
        n_samples, n_states) or (n_samples, n_states).

    Returns
    -------
    partial_covariances : np.ndarray
        Matrix of partial covariance (multiple regression parameter estimates,
        :math:`\beta`). Shape is (n_states, n_channels).

    Note
    ----

    - The regression is done separately for each channel.
    - State/mode time courses are variance normalized so that all amplitude
      info goes into the partial covariances, :math:`\beta_i`.
    """
    if type(data) != type(alpha):
        raise ValueError(
            "data and alpha must be the same type: numpy arrays or lists of "
            "numpy arrays."
        )
    if isinstance(data, np.ndarray):
        data = [data]
        alpha = [alpha]
    for i in range(len(data)):
        if data[i].shape[0] != alpha[i].shape[0]:
            raise ValueError("Difference number of samples in data and alpha.")

    pcovs = []
    for X, a in zip(data, alpha):
        # Variance normalise state/mode time courses
        a_normed = a / np.std(a, axis=0, keepdims=True)

        # Do multiple regression of alpha onto data
        pcovs.append(np.linalg.pinv(a_normed) @ X)

    return np.squeeze(pcovs)


def ae_hmm_networks(data, alpha):
    """Calculate subject-specific AE-HMM networks.

    Parameters
    ----------
    data : list of np.ndarray
        Amplitude envelope data (i.e. after preparation).
        Shape is (n_subjects, n_channels).
    alpha : list of np.ndarray
        Inferred state probabilities. Shape is (n_subjects, n_states).

    Returns
    -------
    means : np.ndarray
        Subject-specific mean activity maps.
        Shape is (n_subjects, n_states, n_channels).
    aecs : np.ndarray
        Subject-specific amplitude envelope correlation maps.
        Shape is (n_subjects, n_states, n_channels, n_channels).
    """
    if type(data) != type(alpha):
        raise TypeError(
            "data and alpha must be the same type: numpy arrays or lists of "
            "numpy arrays."
        )
    if isinstance(data, np.ndarray):
        data = [data]
        alpha = [alpha]

    # Hard classify the state probabilties
    stc = inference.modes.argmax_time_courses(alpha)

    # Number of subjects, channels and states
    n_subjects = len(data)
    n_channels = data[0].shape[1]
    n_states = stc[0].shape[1]

    # Calculate subject and state specific networks
    _logger.info("Calculating AE networks")
    means = np.empty([n_subjects, n_states, n_channels], dtype=np.float32)
    aecs = np.empty([n_subjects, n_states, n_channels, n_channels], dtype=np.float32)
    for i in range(n_subjects):
        d = data[i][: stc[i].shape[0]]
        for j in range(n_states):
            x = d[stc[i][:, j] == 1]
            means[i, j] = np.mean(x, axis=0)
            aecs[i, j] = np.corrcoef(x, rowvar=False)

    return means, aecs
