"""Functions to manipulate and analyse inferred mode/state data.

"""

import numpy as np

from osl_dynamics import array_ops


def autocorrelation_functions(
    mode_covariances,
    n_embeddings,
    pca_components,
):
    """Auto/cross-correlation function from the mode covariance matrices.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariance matrices.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.

    Returns
    -------
    acfs : np.ndarray
        Auto/cross-correlation functions.
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

    # Dimensions
    n_subjects = te_covs.shape[0]
    n_modes = te_covs.shape[1]
    n_parcels = te_covs.shape[-1] // n_embeddings
    n_acf = 2 * n_embeddings - 1

    # Take mean of elements from the time embedded covariances that
    # correspond to the auto/cross-correlation function
    blocks = te_covs.reshape(
        n_subjects, n_modes, n_parcels, n_embeddings, n_parcels, n_embeddings
    )
    acfs = np.empty([n_subjects, n_modes, n_parcels, n_parcels, n_acf])
    for i in range(n_acf):
        acfs[:, :, :, :, i] = np.mean(
            np.diagonal(blocks, offset=i - n_embeddings + 1, axis1=3, axis2=5), axis=-1
        )

    return np.squeeze(acfs)


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
    zero_lag : bool
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
            :, :, n_embeddings // 2 :: n_embeddings, n_embeddings // 2 :: n_embeddings
        ]

    else:
        # Return block means
        n_subjects = te_covs.shape[0]
        n_modes = te_covs.shape[1]
        n_parcels = te_covs.shape[-1] // n_embeddings

        n_parcels = te_covs.shape[-1] // n_embeddings
        blocks = te_covs.reshape(
            n_subjects, n_modes, n_parcels, n_embeddings, n_parcels, n_embeddings
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


def partial_covariances(data, alpha):
    """Calculate partial covariances.

    Returns the multiple regression parameters estimates, pcovs, of the state
    time courses regressed onto the data from each channel. The regression is
    done separately for each channel. I.e. pcovs is the estimate of the
    (n_modes, n_channels) matrix, Beta. We fit the regression Y_i = X @ Beta_i + e,
    where:

    - Y_i is (n_samples, 1) the data amplitude/envelope/power/abs time course at channel i.
    - X is (n_samples, n_modes) matrix of the variance normalised mode time courses (i.e. alpha).
    - Beta_i is (n_modes, 1) vector of multiple regression parameters for channel i.
    - e is the error.

    NOTE: state time courses are variance normalised so that all amplitude info goes
    into Beta (i.e. pcovs).

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        Training data for each subject.
    alpha : np.ndarray or list of np.ndarray
        State/mode time courses for each subject.

    Returns
    -------
    pcovs : np.ndarray
        Matrix of partial covariance (multiple regression parameter estimates).
        Shape is (n_modes, n_channels).
    """
    if type(data) != type(alpha):
        raise ValueError(
            "data and alpha must be the same type: numpy arrays or lists of numpy arrays."
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
        a /= np.std(a, axis=0, keepdims=True)

        # Do multiple regression of alpha onto data
        pcovs.append(np.linalg.pinv(a) @ X)

    return np.squeeze(pcovs)


def state_activation(state_time_course):
    """Calculate state activations from a state time course.

    Given a state time course (strictly binary), calculate the beginning and
    end of each activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    ons : list of numpy.ndarray
        List containing state beginnings (index) in the order they occur for
        each state. This cannot necessarily be converted into an array as an
        equal number of elements in each array is not guaranteed.
    offs : list of numpy.ndarray
        List containing state ends (index) in the order they occur for each state.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.
    """
    state_on = []
    state_off = []

    diffs = np.diff(state_time_course, axis=0)
    for i, diff in enumerate(diffs.T):
        on = (diff == 1).nonzero()[0]
        off = (diff == -1).nonzero()[0]
        try:
            if on[-1] > off[-1]:
                off = np.append(off, len(diff))

            if off[0] < on[0]:
                on = np.insert(on, 0, -1)

            state_on.append(on)
            state_off.append(off)
        except IndexError:
            print(f"No activation in state {i}.")
            state_on.append(np.array([]))
            state_off.append(np.array([]))

    state_on = np.array(state_on, dtype=object)
    state_off = np.array(state_off, dtype=object)

    return state_on, state_off


def lifetimes(state_time_course, sampling_frequency=None):
    """Calculate state lifetimes from a state time course.

    Given a state time course (one-hot encoded), calculate the lifetime of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, or n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    lts : list of numpy.ndarray
        List containing an array of lifetimes in the order they occur for each
        state. This cannot necessarily be converted into an array as an equal
        number of elements in each array is not guaranteed. Shape is (n_subjects,
        n_states, n_activations) or (n_states, n_activations).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]
    lts = []
    for stc in state_time_course:
        ons, offs = state_activation(stc)
        lt = offs - ons
        if sampling_frequency is not None:
            lt = [lt_ / sampling_frequency for lt_ in lt]
        lts.append(lt)
    if len(lts) == 1:
        lts = lts[0]
    return lts


def lifetime_statistics(state_time_course, sampling_frequency=None):
    """Calculate statistics of the lifetime distribution of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    means : np.ndarray
        Mean lifetime of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    std : np.ndarray
        Standard deviation of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    """
    lts = lifetimes(state_time_course, sampling_frequency)
    if np.array(lts, dtype=object).ndim == 2:
        # lts.shape = (n_subjects, n_states, n_activations)
        mean = []
        std = []
        for i in range(len(lts)):
            mean.append([np.mean(lt) for lt in lts[i]])
            std.append([np.std(lt) for lt in lts[i]])
    else:
        # lts.shape = (n_states, n_activations)
        mean = [np.mean(lt) for lt in lts]
        std = [np.std(lt) for lt in lts]
    return np.array(mean), np.array(std)


def mean_lifetimes(state_time_course, sampling_frequency=None):
    """Calculate the mean lifetime of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    mlt : np.ndarray
        Mean lifetime of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    """
    return lifetime_statistics(state_time_course, sampling_frequency)[0]


def intervals(state_time_course, sampling_frequency=None):
    """Calculate state intervals from a state time course.

    An interval is the duration between successive visits for a particular state.

    Parameters
    ----------
    state_time_course : list or numpy.ndarray
        State time course (strictly binary). Shape must be (n_subjects, n_samples,
        n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the intervals in seconds.

    Returns
    -------
    intvs : list of numpy.ndarray
        List containing an array of intervals in the order they occur for each
        state. This cannot necessarily be converted into an array as an equal
        number of elements in each array is not guaranteed. Shape is (n_subjects,
        n_states, n_activations) or (n_states, n_activations).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]
    intvs = []
    for stc in state_time_course:
        intv = []
        ons, offs = state_activation(stc)
        for on, off in zip(ons, offs):
            intv.append(on[1:] - off[:-1])
        if sampling_frequency is not None:
            intv = [i / sampling_frequency for i in intv]
        intvs.append(intv)
    if len(intvs) == 1:
        intvs = intvs[0]
    return intvs


def interval_statistics(state_time_course, sampling_frequency=None):
    """Calculate statistics of the interval distribution of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the lifetimes in seconds.

    Returns
    -------
    means : np.ndarray
        Mean interval of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    std : np.ndarray
        Standard deviation of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    """
    intvs = intervals(state_time_course, sampling_frequency)
    if np.array(intvs, dtype=object).ndim == 2:
        # intvs.shape = (n_subjects, n_states, n_activations)
        mean = []
        std = []
        for i in range(len(intvs)):
            mean.append([np.mean(lt) for lt in intvs[i]])
            std.append([np.std(lt) for lt in intvs[i]])
    else:
        # intvs.shape = (n_states, n_activations)
        mean = [np.mean(lt) for lt in intvs]
        std = [np.std(lt) for lt in intvs]
    return np.array(mean), np.array(std)


def mean_intervals(state_time_course, sampling_frequency=None):
    """Calculate the mean interval of each state.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If passed returns the intervals in seconds.

    Returns
    -------
    mlt : np.ndarray
        Mean interval of each state. Shape is (n_subjects, n_states) or
        (n_states,).
    """
    return interval_statistics(state_time_course, sampling_frequency)[0]


def fractional_occupancies(state_time_course):
    """Calculates the fractional occupancy.

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
    if isinstance(state_time_course, list):
        fo = [np.sum(stc, axis=0) / stc.shape[0] for stc in state_time_course]
    else:
        fo = np.sum(state_time_course, axis=0) / state_time_course.shape[0]
    return np.array(fo, dtype=np.float32)


def switching_rates(state_time_course, sampling_frequency=None):
    """Calculates the switching rate.

    This is defined as the number of state activations per second.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    sampling_frequency : float
        Sampling frequency in Hz. If None, defaults to 1 Hz.

    Returns
    -------
    sr : np.ndarray
        The switching rate of each state. Shape is (n_subjects, n_states)
        or (n_states,).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]

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


def fano_factor(
    state_time_course,
    window_lengths,
    sampling_frequency=1.0,
):
    """Calculates the Fano factor.

    Parameters
    ----------
    state_time_course : list or np.ndarray
        State time course (strictly binary). Shape must be (n_subjects,
        n_samples, n_states) or (n_samples, n_states).
    window_lengths : list or np.ndarray
        Window lengths to use. Must be in samples.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    F : list of np.ndarray
        Fano factor. Shape is (n_subjects, n_window_lengths, n_states)
        or (n_window_lengths, n_states).
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
        State time courses.
        Shape must be (n_subjects, n_samples, n_states) or (n_samples, n_states).
    n_states : int
        Number of states.

    Returns
    -------
    trans_prob : np.ndarray
        Subject-specific transition probability matrices.
        Shape is (n_subjects, n_states, n_states).
    """
    if isinstance(state_time_course, np.ndarray):
        state_time_course = [state_time_course]
    trans_prob = []
    for stc in state_time_course:
        stc = stc.argmax(axis=1)
        vals, counts = np.unique(
            stc[np.arange(2)[None, :] + np.arange(len(stc) - 1)[:, None]],
            axis=0,
            return_counts=True,
        )
        if n_states is None:
            n_states = stc.max() + 1
        tp = np.zeros((n_states, n_states))
        tp[vals[:, 0], vals[:, 1]] = counts
        with np.errstate(divide="ignore", invalid="ignore"):
            tp /= tp.sum(axis=1)[:, None]
        trans_prob.append(np.nan_to_num(tp))
    return np.squeeze(trans_prob)


def simple_moving_average(data, window_length, step_size):
    """Simple moving average.

    Calculates moving averages by computing the unweighted mean of n
    observations over the current time window. Can be used to smooth
    the fractional occupancy time courses as in Baker et al. (2014).

    Parameters
    ----------
    data: np.ndarray
        Time series data with a shape (n_samples, n_modes).
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
        :, window_length // 2 : window_length // 2 + n_modes
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
