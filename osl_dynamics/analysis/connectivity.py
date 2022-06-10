"""Functions to calculate and plot network connectivity.

"""

from pathlib import Path

import numpy as np
from nilearn import plotting
from nilearn.plotting.cm import _cmap_d as cm
from tqdm import trange
from osl_dynamics import array_ops
from osl_dynamics.analysis.gmm import fit_gaussian_mixture
from osl_dynamics.analysis.spectral import get_frequency_args_range
from osl_dynamics.utils.parcellation import Parcellation
from osl_dynamics.utils.misc import override_dict_defaults


def covariance_from_spectra(
    frequencies,
    power_spectra,
    components=None,
    frequency_range=None,
):
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    power_spectra : np.ndarray
        Power/cross spectra for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Default is full range.

    Returns
    -------
    covar : np.ndarray
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
            for j in range(n_components):
                c[j] /= np.sum(components[j])

        else:
            # Integrate over the given frequency range
            if frequency_range is None:
                c = np.sum(csd, axis=-1)
            else:
                [f_min, f_max] = get_frequency_args_range(frequencies, frequency_range)
                c = np.sum(csd[..., f_min:f_max], axis=-1)

        c = c.reshape(n_components, n_modes, n_channels, n_channels)
        covar.append(c)

    return np.squeeze(covar)


def mean_coherence_from_spectra(
    frequencies,
    coherence,
    components=None,
    frequency_range=None,
    fit_gmm=False,
    gmm_filename=None,
):
    """Calculates mean coherence from spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    coherence : np.ndarray
        Coherence for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz).
    fit_gmm : bool
        Should we fit a two component Gaussian mixture model and only keep
        one of the components.
    gmm_filename : str
        Filename to save GMM plot. Only used if fit_gmm=True.

    Returns
    -------
    coh : np.ndarray
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
            for j in range(n_components):
                coh[j] /= np.sum(components[j])

        else:
            # Mean over the given frequency range
            if frequency_range is None:
                coh = np.mean(coh, axis=-1)
            else:
                [f_min, f_max] = get_frequency_args_range(frequencies, frequency_range)
                coh = np.mean(coh[..., f_min:f_max], axis=-1)

        coh = coh.reshape(n_components, n_modes, n_channels, n_channels)
        c.append(coh)

    if fit_gmm:
        # Mean coherence over modes
        mean_coh = np.mean(coh, axis=1)

        # Indices for off diagonal elements
        m, n = np.triu_indices(n_channels, k=1)

        # Loop over components and modes
        for i in range(n_components):
            for j in range(n_modes):

                # Off diagonal coherence values to fit a GMM to
                c = coh[i, j, m, n] - mean_coh[i, m, n]

                # Replace nans with mean value so that they don't affect the GMM fit
                c[np.isnan(c)] = np.mean(c[~np.isnan(c)])

                # Fit a GMM
                if gmm_filename is not None:
                    plot_filename = (
                        "{fn.parent}/{fn.stem}{i:0{w1}d}_{j:0{w2}d}{fn.suffix}".format(
                            fn=Path(gmm_filename),
                            i=i,
                            j=j,
                            w1=len(str(n_components)),
                            w2=len(str(n_modes)),
                        )
                    )
                else:
                    plot_filename = None
                mixture_label = fit_gaussian_mixture(
                    c,
                    print_message=False,
                    plot_filename=plot_filename,
                    bayesian=True,
                    max_iter=5000,
                    n_init=10,
                )

                # Only keep the second mixture component and remove nan connections
                c = coh[i, j, m, n]
                c[mixture_label == 0] = 0
                c[np.isnan(c)] = 0
                coh[i, j, m, n] = c
                coh[i, j, n, m] = c

    return np.squeeze(coh)


def save(
    connectivity_map,
    threshold,
    filename,
    parcellation_file,
    component=None,
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
        Spectral component to save.
    plot_kwargs : dict
        Keyword arguments to pass to nilearn.plotting.plot_connectome.
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

    # Load parcellation file
    parcellation = Parcellation(parcellation_file)

    # Select the component we're plotting
    conn_map = connectivity_map[component]

    # Default plotting settings
    default_plot_kwargs = {
        "node_size": 10,
        "node_color": "black",
        "edge_cmap": cm["red_transparent_full_alpha_range"],
        "colorbar": True,
    }

    # Overwrite keyword arguments if passed
    plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    # Plot maps
    n_modes = conn_map.shape[0]
    for i in trange(n_modes, desc="Saving images", ncols=98):
        output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
            fn=Path(filename), i=i, w=len(str(n_modes))
        )
        plotting.plot_connectome(
            conn_map[i],
            parcellation.roi_centers(),
            edge_threshold=f"{threshold * 100}%",
            output_file=output_file,
            **plot_kwargs,
        )
