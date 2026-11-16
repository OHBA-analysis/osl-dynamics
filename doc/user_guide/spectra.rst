Spectral Analysis
=================

After training a model on time-delay embedded (TDE) data, we describe each state or mode by its power spectral density (PSD) and coherence. These are calculated from the parcel time courses before time-delay embedding (and PCA), using the inferred state probabilities or mixing coefficients (:code:`alpha`).

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Function
     - Models
     - Returns
   * - :func:`~osl_dynamics.analysis.spectral.regression_spectra`
     - HMM, DyNeMo
     - The static spectrum and the change from the static spectrum for each state/mode
   * - :func:`~osl_dynamics.analysis.spectral.multitaper_spectra`
     - HMM
     - The spectrum of each state

For examples with real data, see the :doc:`HMM Multitaper Spectra </tutorials_build/4-1_hmm_multitaper_spectra>` and :doc:`HMM Plotting MEG Networks </tutorials_build/4-2_hmm_plotting_meg_networks>` tutorials (multitaper), and the :doc:`DyNeMo Regression Spectra </tutorials_build/5-1_dynemo_regression_spectra>` and :doc:`DyNeMo Plotting Networks </tutorials_build/5-2_dynemo_plotting_networks>` tutorials (regression).

Regression spectra
------------------

The spectrum at time :math:`t` is modeled as a mixture of the spectra of each state/mode:

.. math::
    P_t(f) = \sum_j \alpha_{jt} P_j(f).

:func:`~osl_dynamics.analysis.spectral.regression_spectra` calculates a spectrogram :math:`P_t` (cross spectra in a sliding window of :code:`window_length` samples, each split into :code:`n_sub_windows` Hann-tapered sub-windows), averages :code:`alpha` over the same windows and fits the mixture with least squares, i.e. a GLM-spectrum [1]_. The frequency resolution is set by the sub-window length, :code:`window_length / n_sub_windows` samples.

Definitions
***********

**Static spectrum** :math:`S`. The time-averaged spectrum of the data, :math:`S = \sum_j \bar{\alpha}_j P_j`, where :math:`\bar{\alpha}_j` is the mean of :code:`alpha` for mode :math:`j`.

**Mode spectrum** :math:`P_j`. The spectrum when only state/mode :math:`j` is active. For the HMM, this is the spectrum of the state. For DyNeMo, the mixing coefficients of a mode rarely approach one, so :math:`P_j` is an extrapolation that doesn't typically arise.

**Standard deviation** :math:`\sigma_j`. The standard deviation of the windowed :code:`alpha` for mode :math:`j`, averaged over sessions. It is the typical size of the fluctuations in a mode's :code:`alpha` over the window length.

Outputs
*******

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Output
     - Value
     - Interpretation
   * - Intercept, :code:`psd[:, 1]` with :code:`return_coef_int=True`
     - :math:`S`
     - The static spectrum.
   * - Coefficients, :code:`psd[:, 0]` with :code:`return_coef_int=True`
     - :math:`\beta_j = \sigma_j (P_j - S)`
     - The change in the spectrum from the static spectrum for a one standard deviation increase in mode :math:`j`'s :code:`alpha`.
   * - :code:`psd` with :code:`return_coef_int=False`
     - :math:`S + \beta_j`
     - The spectrum when a mode's :code:`alpha` is one standard deviation above its mean.
   * - :code:`sigma` with :code:`return_sigma=True`
     - :math:`\sigma_j`
     - Gives the mode spectrum, :math:`P_j = S + \beta_j / \sigma_j`.
   * - :code:`coh`
     - :math:`C_j`
     - The change in the cross spectrum, normalized by the static power (change in coherence), for a one standard deviation increase in mode :math:`j`'s :code:`alpha`.

The coherence is the change in the cross spectrum in the direction of the static cross spectrum, normalized by the static power (change in coherence):

.. math::
    C_{jxy} = \frac{\text{Re}(\beta_{jxy} e^{-i\phi_{xy}})}{\sqrt{S_{xx} S_{yy}}},

where :math:`\phi_{xy}` is the phase of :math:`S_{xy}`. It includes changes in both coherence and power: it increases when the coherence between :math:`x` and :math:`y` increases or when the power at :math:`x` or :math:`y` increases. The diagonal is the relative change in power, :math:`\beta_{jxx} / S_{xx}`.

Positive coefficients mean a mode has more power than the static spectrum and negative coefficients mean less. The changes of all modes balance around the static spectrum, so some modes can be negative everywhere: e.g. a DyNeMo mode with a small covariance that takes most of the mixing coefficients.

Sessions
********

The coefficients of every session are scaled by the same :math:`\sigma_j`, so session-specific spectra have the same units and can be averaged (with the weights :code:`w` from :code:`return_weights=True`) or compared with :doc:`statistical tests </user_guide/stats>`.

- Calculate the spectra for all the sessions you want to compare in one call. To compare spectra from separate calls or different window settings, divide the coefficients by :math:`\sigma_j` (:code:`return_sigma=True`).
- Session-specific estimates are noisier for sessions where a mode's :code:`alpha` varies less than :math:`\sigma_j`.
- A mode whose :code:`alpha` is (nearly) constant in a session (e.g. an HMM state that is never visited) has NaN coefficients. Average over sessions with :code:`np.ma.average(np.ma.masked_invalid(psd), axis=0, weights=w)`.

Multitaper spectra
------------------

:func:`~osl_dynamics.analysis.spectral.multitaper_spectra` multiplies the data by each state's probability time course and calculates a multitaper spectrum, which estimates the state spectrum :math:`P_j` directly. It is the recommended method for HMM states (see the :doc:`HMM page </user_guide/models/hmm>`). The multitaper spectra have broader peaks and smaller differences between states than the regression spectra.

To compare the two methods, calculate :math:`P_j = S + \beta_j / \sigma_j` from the regression spectra, or the change from the static spectrum, :math:`P_j - S`, from the multitaper spectra, where :math:`S` is the average of the state spectra weighted by the fractional occupancies.

What to plot
------------

For regression spectra, plot the coefficients and coherences directly, without subtracting the mean over modes.

.. code-block:: python

    import numpy as np
    from osl_dynamics.analysis import spectral, power, connectivity
    from osl_dynamics.utils import plotting

    f, psd, coh, w, sigma = spectral.regression_spectra(
        data=data,  # parcel time courses (before TDE/PCA)
        alpha=alpha,  # state probabilities or mixing coefficients
        sampling_frequency=250,
        window_length=1000,
        step_size=20,
        n_sub_windows=8,
        frequency_range=[1, 45],
        return_coef_int=True,
        return_weights=True,
        return_sigma=True,
    )

PSDs
****

There are two options:

#. **Without the static PSD**: the coefficients, :math:`\beta_j`, which show the frequencies where a mode has more (positive) or less (negative) power than the static PSD.
#. **With the static PSD**: the static PSD, :math:`S`, together with the PSD of each mode. For DyNeMo, use :math:`S + \beta_j`, the PSD when a mode's :code:`alpha` is one standard deviation above its mean. For the HMM, use :math:`P_j = S + \beta_j / \sigma_j`, the PSD of each state (the quantity estimated by the multitaper).

.. code-block:: python

    # Group average, averaged over parcels
    gpsd = np.average(psd, axis=0, weights=w)  # (2, modes, parcels, freq)
    coefs = gpsd[0].mean(axis=1)  # (modes, freq)
    static = gpsd[1, 0].mean(axis=0)  # (freq,)
    n_modes = coefs.shape[0]
    labels = [f"Mode {j}" for j in range(n_modes)]

    # Option 1: without the static PSD (change from the static PSD)
    plotting.plot_line(
        [f] * n_modes,
        coefs,
        labels=labels,
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        filename="psd_change.png",
    )

    # Option 2: with the static PSD
    mode_psd = static + coefs  # DyNeMo
    # mode_psd = static + coefs / sigma[:, np.newaxis]  # HMM
    plotting.plot_line(
        [f] * (n_modes + 1),
        [static, *mode_psd],
        labels=["Static"] + labels,
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        filename="psd_static_and_modes.png",
    )

For DyNeMo, :math:`S + \beta_j` is close to the static PSD, so the differences between modes are clearer without the static PSD. Use :code:`gpsd[0][:, i]` and :code:`gpsd[1, 0, i]` instead of the mean over parcels to plot parcel :code:`i`. Avoid plotting :math:`P_j` for DyNeMo.

Power maps and coherence networks
*********************************

.. code-block:: python

    coefs = psd[:, 0]  # (sessions, modes, parcels, freq)

    # Power maps
    p = power.variance_from_spectra(f, coefs)
    p = np.average(p, axis=0, weights=w)
    power.save(p, parcellation_file=parcellation_file, filename="pow_.png")

    # Coherence networks: the 2% of edges with the largest changes
    c = connectivity.mean_coherence_from_spectra(f, coh)
    c = np.average(c, axis=0, weights=w)
    c_thres = connectivity.threshold(c, percentile=98, absolute_value=True)
    connectivity.save(c_thres, parcellation_file=parcellation_file, filename="coh_.png")

- **Power maps.** The coefficients integrated over frequency with :func:`~osl_dynamics.analysis.power.variance_from_spectra`: the change in power from the static power, as a fraction of each parcel's variance (with the default :code:`standardize=True`). Red is an increase and blue a decrease. Pass :code:`frequency_range` (e.g. :code:`[8, 12]`) for a frequency band. Divide by the intercept for the relative change in power. Use the coefficients, not :math:`S + \beta_j`, for power maps and to compare modes.
- **Coherence networks.** The edges with the largest changes in coherence, increases (red) and decreases (blue).

References
----------

.. [1] Quinn, A. J. et al. (2024). The GLM-spectrum: A multilevel framework for spectrum analysis with covariate and confound modelling. `Imaging Neuroscience <https://doi.org/10.1162/imag_a_00082>`_, 2.
