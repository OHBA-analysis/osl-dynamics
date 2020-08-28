"""Functions to generate maps.

"""

import numpy as np
from vrad.analysis.functions import validate_array


def state_maps(power_spectra, coherences, components):
    """Calculates spatial maps for each spectral component and state."""

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_frequency_bins) "
        + "or 4D numpy array (n_states, n_channels, n_channels, "
        + "n_frequency_bins) must be passed for spectra."
    )
    power_spectra = validate_array(
        power_spectra,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )
    coherences = validate_array(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    # Number of subjects, states, channels and frequency bins
    n_subjects, n_states, n_channels, n_channels, n_f = power_spectra.shape

    # Number of components
    n_components = components.shape[0]

    # Remove cross-spectral densities from the power spectra array and concatenate
    # over subjects and states
    psd = power_spectra[:, :, range(n_channels), range(n_channels)].reshape(-1, n_f)

    # PSDs are real valued so we can recast
    psd = psd.real

    # Calculate PSDs for each spectral component
    psd = components @ psd.T
    psd = psd.reshape(n_components, n_states, n_channels)

    # Power map
    p = np.zeros([n_components, n_states, n_channels, n_channels])
    p[:, :, range(n_channels), range(n_channels)] = psd

    # Only keep the upper triangle of the coherences and concatenate over subjects
    # and states
    i, j = np.triu_indices(n_channels, 1)
    coh = coherences[:, :, i, j].reshape(-1, n_f)

    #  Calculate coherences for each spectral component
    coh = components @ coh.T
    coh = coh.reshape(n_components, n_states, n_channels * (n_channels - 1) // 2)

    # Coherence map
    c = np.zeros([n_components, n_states, n_channels, n_channels])
    c[:, :, i, j] = coh
    c[:, :, j, i] = coh
    c[:, :, range(n_channels), range(n_channels)] = 1

    return p, c
