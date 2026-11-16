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

See the :doc:`DyNeMo Regression Spectra </tutorials_build/5-1_dynemo_regression_spectra>` and :doc:`DyNeMo Plotting Networks </tutorials_build/5-2_dynemo_plotting_networks>` tutorials for examples with real data.

Regression spectra
------------------

The spectrum at time :math:`t` is modeled as a mixture of the spectra of each state/mode:

.. math::
    P_t(f) = \sum_j \alpha_{jt} P_j(f).

:func:`~osl_dynamics.analysis.spectral.regression_spectra` calculates a spectrogram :math:`P_t` (cross spectra in a sliding window of :code:`window_length` samples, each split into :code:`n_sub_windows` Hann-tapered sub-windows), averages :code:`alpha` over the same windows and fits the mixture with least squares. The frequency resolution is set by the sub-window length, :code:`window_length / n_sub_windows` samples.

For DyNeMo, pass the mixing coefficients returned by the model. The mixing coefficients re-weighted with :func:`~osl_dynamics.inference.modes.reweight_alphas` are for summary statistics.

Definitions
***********

**Static spectrum** :math:`S`. The time-averaged spectrum of the data, :math:`S = \sum_j \bar{\alpha}_j P_j`, where :math:`\bar{\alpha}_j` is the mean of :code:`alpha` for mode :math:`j`.

**Mode spectrum** :math:`P_j`. The spectrum when only state/mode :math:`j` is active. For the HMM, this is the spectrum of the state. For DyNeMo, the mixing coefficients of a mode rarely approach one, so :math:`P_j` is an extrapolation and should not be interpreted.

**Standard deviation** :math:`\sigma_j`. The standard deviation of the windowed :code:`alpha` for mode :math:`j`, averaged over sessions. It is the typical size of the fluctuations in a mode's :code:`alpha` over the window length. For the HMM, it is the typical change in the fraction of a window spent in a state. :math:`\sigma_j` decreases with longer windows and depends on the sessions passed.

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
     - The change in coherence from the static coherence for a one standard deviation increase in mode :math:`j`'s :code:`alpha`.

The coherence is the change in the cross spectrum in the direction of the static cross spectrum, normalized by the static power:

.. math::
    C_{jxy} = \frac{\text{Re}(\beta_{jxy} e^{-i\phi_{xy}})}{\sqrt{S_{xx} S_{yy}}},

where :math:`\phi_{xy}` is the phase of :math:`S_{xy}`. It is positive where the coherence increases and negative where it decreases. The diagonal is the relative change in power, :math:`\beta_{jxx} / S_{xx}`.

Positive coefficients mean a mode has more power than the static spectrum and negative coefficients mean less. The changes of all modes balance around the static spectrum, so some modes can be negative everywhere: e.g. a DyNeMo mode with a small covariance that takes most of the mixing coefficients.

Sessions
********

The coefficients of every session are scaled by the same :math:`\sigma_j`, so session-specific spectra have the same units and can be averaged (with the weights :code:`w` from :code:`return_weights=True`) or compared with :doc:`statistical tests </user_guide/stats>`.

- Calculate the spectra for all the sessions you want to compare in one call. To compare spectra from separate calls or different window settings, divide the coefficients by :math:`\sigma_j` (:code:`return_sigma=True`).
- Session-specific estimates are noisier for sessions where a mode's :code:`alpha` varies less than :math:`\sigma_j`.
- A mode with a constant :code:`alpha` in a session (e.g. an HMM state that is never visited) has NaN coefficients. Average over sessions with :code:`np.ma.average(np.ma.masked_invalid(psd), axis=0, weights=w)`.

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

    f, psd, coh, w = spectral.regression_spectra(
        data=data,  # parcel time courses (before TDE/PCA)
        alpha=alpha,  # state probabilities or mixing coefficients
        sampling_frequency=250,
        window_length=1000,
        step_size=20,
        n_sub_windows=8,
        frequency_range=[1, 45],
        return_coef_int=True,
        return_weights=True,
    )
    coefs = psd[:, 0]  # (sessions, modes, parcels, freq)

    # PSDs: group average, averaged over parcels
    gpsd = np.average(coefs, axis=0, weights=w).mean(axis=1)

    # Power maps
    p = power.variance_from_spectra(f, coefs)
    p = np.average(p, axis=0, weights=w)
    power.save(p, parcellation_file=parcellation_file, filename="pow_.png")

    # Coherence networks: the 2% of edges with the largest changes
    c = connectivity.mean_coherence_from_spectra(f, coh)
    c = np.average(c, axis=0, weights=w)
    c_thres = connectivity.threshold(c, percentile=98, absolute_value=True)
    connectivity.save(c_thres, parcellation_file=parcellation_file, filename="coh_.png")

    # Coherence maps: mean change in coherence of each parcel's edges
    n_parcels = c.shape[-1]
    c[:, range(n_parcels), range(n_parcels)] = 0
    power.save(c.sum(axis=-1) / (n_parcels - 1), parcellation_file=parcellation_file, filename="coh_map_.png")

- **PSDs.** The coefficients, which show the frequencies where a mode has more or less power than the static spectrum.
- **Power maps.** The coefficients integrated over frequency with :func:`~osl_dynamics.analysis.power.variance_from_spectra`: the change in power from the static power, as a fraction of each parcel's variance (with the default :code:`standardize=True`). Red is an increase and blue a decrease. Pass :code:`frequency_range` (e.g. :code:`[8, 12]`) for a frequency band. Divide by the intercept for the relative change in power.
- **Coherence networks.** The edges with the largest changes in coherence, increases (red) and decreases (blue).
- **HMM state spectra.** To compare with the multitaper or across window settings, plot :math:`P_j - S = \beta_j / \sigma_j` (:code:`return_sigma=True`).
- Avoid plotting :code:`psd` with :code:`return_coef_int=False`, which is dominated by the static spectrum, and :math:`P_j` for DyNeMo.
