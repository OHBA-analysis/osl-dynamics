"""Functions to calculate/plot connectivity.

"""

from pathlib import Path

import numpy as np
from nilearn import plotting
from tqdm import trange
from vrad import array_ops
from vrad.utils.parcellation import Parcellation


def covariance_from_spectra(
    frequencies: np.ndarray,
    power_spectra: np.ndarray,
    components: np.ndarray = None,
    frequency_range: list = None,
) -> np.ndarray:
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    power_spectra : np.ndarray
        Power/cross spectra for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f). Optional.
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Optional: default is full
        range.

    Returns
    -------
    np.ndarray
        Covariance over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels).
    """

    # Validation
    error_message = (
        "A (n_channels, n_channels, n_frequency_bins), "
        + "(n_modes, n_channels, n_channels, n_frequency_bins) or "
        + "(n_subjects, n_modes, n_channels, n_channels, n_frequency_bins) "
        + "array must be passed."
    )
    power_spectra = array_ops.validate(
        power_spectra,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_subjects, n_modes, n_channels, n_channels, n_f = power_spectra.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity maps for each subject
    covar = []
    for i in range(n_subjects):
        # Cross spectral densities
        csd = power_spectra[i].reshape(-1, n_f)
        csd = abs(csd)

        if components is not None:
            # Calculate PSD for each spectral component
            c = components @ csd.T

        else:
            # Integrate over the given frequency range
            if frequency_range is None:
                c = np.sum(csd, axis=-1)
            else:
                f_min_arg = np.argwhere(frequencies > frequency_range[0])[0, 0]
                f_max_arg = np.argwhere(frequencies < frequency_range[1])[-1, 0]
                if f_max_arg < f_min_arg:
                    raise ValueError("Cannot select the specified frequency range.")
                c = np.sum(csd[..., f_min_arg : f_max_arg + 1], axis=-1)

        c = c.reshape(n_components, n_modes, n_channels, n_channels)
        covar.append(c)

    return np.squeeze(covar)


def mean_coherence_from_spectra(
    frequencies: np.ndarray,
    coherence: np.ndarray,
    components: np.ndarray = None,
    frequency_range: list = None,
) -> np.ndarray:
    """Calculates mean coherence from spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    coherence : np.ndarray
        Coherence for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f). Optional.
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Optional: default is full
        range.

    Returns
    -------
    np.ndarray
        Mean coherence over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels).
    """

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_frequency_bins) "
        + "or 4D numpy array (n_modes, n_channels, n_channels, "
        + "n_frequency_bins) must be passed for spectra."
    )
    coherence = array_ops.validate(
        coherence,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_subjects, n_modes, n_channels, n_channels, n_f = coherence.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity for each subject
    c = []
    for i in range(n_subjects):

        # Concatenate over modes
        coh = coherence[i].reshape(-1, n_f)

        if components is not None:
            # Coherence for each spectral component
            coh = components @ coh.T

        else:
            # Mean over the given frequency range
            if frequency_range is None:
                coh = np.mean(coh, axis=-1)
            else:
                f_min_arg = np.argwhere(frequencies > frequency_range[0])[0, 0]
                f_max_arg = np.argwhere(frequencies < frequency_range[1])[-1, 0]
                if f_max_arg < f_min_arg:
                    raise ValueError("Cannot select the specified frequency range.")
                coh = np.mean(coh[..., f_min_arg : f_max_arg + 1], axis=-1)

        coh = coh.reshape(n_components, n_modes, n_channels, n_channels)
        c.append(coh)

    return np.squeeze(coh)


def save(
    connectivity_map: np.ndarray,
    threshold: float,
    filename: str,
    parcellation_file: str,
    component: int = None,
    **plot_kwargs,
):
    """Save connectivity maps.

    Parameters
    ----------
    connectivity_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_modes, n_channels, n_channels).
    threshold : float
        Threshold to determine which connectivity to show.
        Should be between 0 and 1.
    filename : str
        Output filename.
    parcellation_file : str
        Name of parcellation file used.
    component : int
        Spectral component to save. Optional.
    """
    # Validation
    error_message = (
        "Dimensionality of connectivity_map must be 3 or 4, "
        + f"got ndim={connectivity_map.ndim}."
    )
    connectivity_map = array_ops.validate(
        connectivity_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    if threshold > 1 or threshold < 0:
        raise ValueError("threshold must be between 0 and 1.")

    if component is None:
        component = 0

    parcellation = Parcellation(parcellation_file)

    # Plot maps
    n_modes = connectivity_map.shape[1]
    for i in trange(n_modes, desc="Saving images", ncols=98):
        conn_map = connectivity_map[component, i].copy()
        np.fill_diagonal(conn_map, 0)
        output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
            fn=Path(filename), i=i, w=len(str(n_modes))
        )
        plotting.plot_connectome(
            conn_map,
            parcellation.roi_centers(),
            colorbar=True,
            edge_threshold=f"{threshold * 100}%",
            output_file=output_file,
            **plot_kwargs,
        )
